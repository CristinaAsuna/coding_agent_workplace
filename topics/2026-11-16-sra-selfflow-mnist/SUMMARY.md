# From SRA to Self-Flow — MNIST L2：执行记录

topic：2026-11-16-sra-selfflow-mnist · [计划](PLAN.md) · 状态：blocked（E1 梯度非有限）

**一句话结论**：E1 第一个条件的训练因非有限梯度停止。独立重放捕获到高 loss scale 下的 FP16 梯度溢出；同一批数据以 scale 1 重算的梯度有限，支持缩放溢出是此次事件的原因。

## 结论先行

- 正式 E1 在 single+full 约 3100 步后停止，未得到任何计划中的 5000、10000 或 20000 步评估；E2 未运行。研究预测均未检验。
- 3000 步检查点的学生、EMA 和优化器状态均有限；GradScaler scale 为 131072。
- 诊断重放在 3143 步发现 `output.weight` 一个 Inf 梯度，总 loss 为 0.18131566。同一模型和 batch 下，FP16 scale 1 与 FP32 scale 1 的梯度均有限。
- 原代码在检测到非有限梯度后立即停止，位于 `GradScaler.step/update` 之前，因此没有机会执行 GradScaler 的跳步和降低 scale。修复的短验证已通过；正式续训尚未启动。

## E1：双时间步与注意力消融

**预测**：用户预测 dual+full 明显更好、dual+sep 与 dual+full 相当；agent 预测前者差不多、后者更差但不超过 1.5 倍。单 seed 比较采用预先指定的 10% 阈值。

**结果**：single+full 训练在约 3100 步后中止，其他条件及正式检查点评估未运行。[E1.json](results/E1.json) 的正式指标为空；此前 200 步 [smoke](results/smoke.json) 不能用于检验预测。未生成 E1 正式图。

## E2：训练时跨噪声交互

**预测**：用户预测误差大幅下降；agent 预测 dual+full 下降至少 50%，single+full 幅度更小。

**结果**：因 E1 未产生 20000 步权重，E2 未运行。见 [E2.json](results/E2.json)；预测未检验，也没有 E2 正式图。

## 意外与反例

检查点 scaler 从默认初始 65536 经 2000 步增长为 131072；失败前的总 loss 有限。重放采用 activation checkpoint 减少显存，3100 步 loss 0.17128074 与原日志的 0.17128 在显示精度内吻合，但重算路径不能保证与原运行逐位一致。同一 batch 的 scale 1 对照支持本次重放事件由缩放导致，不能证明所有潜在数值问题已排除。

## 这个 toy 能说明什么、不能说明什么

- 能：此次诊断定位到一个 FP16 大 scale 下的非有限梯度，并检查了相同 batch 的低 scale 和 FP32 对照。
- 不能：验证双时间步、注意力切断或跨噪声交互的研究预测；不能推广到 ImageNet、潜空间或大模型，也不能与论文 FID 比较。单 seed 无法估计训练随机性；预设 10% 阈值未经校准，接近阈值不能下结论。

## 下一步

- 保持正式运行停止；如获批准，可在新登记运行中从原检查点续跑。前向 NaN、梯度 NaN 及 scale 不大于 1 时仍发生 Inf 均保留停止条件。

## 复现

- 原运行编号 `E1-20261116-071540-pc`；数据只读复用 `/mnt/d/Datasets/MNIST/raw/`，检查点保留在该运行的 `outputs/weights/`。诊断脚本 [diagnose.py](code/diagnose.py) 不改写原检查点。
- 环境：pc WSL2，Python `/home/pc/lab/envs/main/bin/python`，torch 2.11.0+cu128、torchvision 0.26.0+cu128、numpy 2.5.4、scipy 1.18.1、matplotlib 3.11.2，RTX 3060。
- 结果：[E1](results/E1.json)、[E2](results/E2.json)、[诊断](results/diagnostic.json)、[修复验证](results/fix_verification.json)、[smoke](results/smoke.json)。

## 执行记录

- 原 4 小时预算因 smoke 外推约 8.31 小时触发停止，用户随后批准提高至 10 小时。正式运行及 smoke 合计 1189.07 秒。
- 独立 CPU 审计测得检查点加载与张量审计 0.84 秒；GPU 重放内部计时 57.67 秒，峰值保留显存 2573205504 字节。为涵盖进程启动、重复审计和计时误差，后续预算保守另预留 90 秒，即累计至少按 1279.07 秒计。
- 因共享 GPU 空闲不足，诊断重放使用 student block 的 activation checkpoint；未停止其他服务、未改 batch 和原始检查点，未正式续训。完整训练结果与正式图均不存在。
- 修复验证从 3000 步重放至 3143 步：Inf 梯度使 scale 从 131072 降为 65536，学生参数与 EMA 均不变，后续 20 次参数更新成功；耗时 61.01 秒，峰值保留显存 1671430144 字节。为涵盖进程开销，预算保守另预留 90 秒，累计至少按 1369.07 秒计。
