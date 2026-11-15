# From SRA to Self-Flow: Data Augmentation or Self-Supervision?：实验计划

topic：2026-11-15-sra-selfflow-augmentation · 发现过程重建：[DISCOVERY.md](DISCOVERY.md) · 批准日期：2026-11-15

## 目标

在一个几秒钟能训完的合成 flow-matching toy 上亲历三件事：(E1) 切断跨噪声注意力后 DTS 的收益是否保留；(E2) 这份收益是否像“数据增强”那样随数据量增大而缩小；(E3) AS 在大 mask 比例下的训练–采样不一致。执行者只需本文件；DISCOVERY.md 仅作背景。

## 冻结预测

| 实验 | 用户预测 | agent 预测 | 推翻条件（针对 agent 预测） |
|---|---|---|---|
| E1 | 双时间步+AS “比不切断更好” | dual+sep 与 dual+full 的 SWD 之差小于 seed 间 min–max 范围；二者均值都不高于 single+full。toy 里 DTS 收益本身可能落在噪声内（约 40% 概率），那样记为“无法判定” | dual+full 明显优于 single+full（均值差 > seed 范围），而 dual+sep 回到 single+sep 的水平 |
| E2 | “收益逐渐变大” | 收益 G(n)=SWD(single)−SWD(dual) 在 n=64 最大，n=4096 时接近 0（|G| 小于 seed 范围） | G 随 n 持平或增大 |
| E3 | “基本不变” | α=0.5 时 AS 的 SWD 至少是 α=0.25 的 1.5 倍；混入 ρ=0.25 整图样本后回落到 α=0.25 水平的 1.2 倍以内 | α=0.5 与 α=0.25 的 SWD 之差小于 seed 范围 |

## 共同设定（三个实验都用，除旋钮外不得改动）

- **数据**（纯合成，不下载）：每条样本是 T=16 个 token、每个 token 2 维，是一条随机椭圆上的 16 个点：`x_i = r·[cos(θ_i+φ), a·sin(θ_i+φ)] + 0.02·ε`，θ_i=2πi/16，r~U(0.5,1.5)，φ~U(0,2π)，a~U(0.3,1)。训练集 n 条（固定生成 seed = 实验 seed），另生成 2000 条 held-out。
- **模型**：Transformer，4 个 block，宽 64，4 头，可学习位置嵌入；token 线性嵌入；**逐 token 的时间步条件**：t 的正弦嵌入 → MLP → 每个 block 与输出层的逐 token adaLN（shift/scale）；输出层零初始化，输出 2 维速度。
- **flow matching**：`x_t=(1−t)x0+t·x1`，x1~N(0,I)，目标 `x1−x0`，逐 token MSE 等权平均；t~U(0,1)。
- **分组**：每条样本随机取恰好 ⌊α·T⌋ 个位置为第 0 组，其余为第 1 组；默认 α=0.25（4 / 12 个 token）。
  - schedule=single：两组共用一个 t；schedule=dual：两组各自独立采样 t，两组共用同一份噪声 x1。
  - attention=full：不加掩码；attention=sep：所有 block 中只允许同组 token 互相注意（`A_ij = [g_i==g_j]`，跨组 logit 置为 −inf）。single+sep 也按同样方式分组。
- **SRA 对齐损失（固定开启）**：EMA 教师（decay 0.99，toy 步数少，故不取论文的 0.9999）；学生第 1 个 block 之后的特征经 head（Linear 64→128、SiLU、Linear 128→64）与教师第 3 个 block 之后的特征做逐 token `1−cos`（沿通道维），教师 stop-grad；教师输入用与学生相同的 x0、x1、分组和注意力模式，各组时间步为学生的 t 减 U(0,0.2) 后截断到 ≥0；总损失 = 速度损失 + 0.5·对齐损失。
- **训练**：AdamW，lr 1e-3，无 weight decay，batch 128（n<128 时有放回采样），1500 步，CPU。
- **采样**：用 EMA 权重，Euler 50 步从 t=1 到 0，全部 token 同一个 t、完整注意力；每个模型用固定噪声 seed 生成 2000 条。
- **主指标 SWD（越低越好）**：把样本展平成 32 维，取固定的 256 个随机方向，逐方向的一维 Wasserstein-1 的平均，生成样本 vs held-out。
- **副指标 记忆比 mem**：生成样本到训练集最近邻距离的中位数 ÷ held-out 样本到训练集最近邻距离的中位数（≈1 泛化，→0 背训练集）。
- **参照线**（每个实验都算并画出）：真实下限 = 两批各 2000 条独立真实样本之间的 SWD；零信息基线 = 逐维匹配均值方差的独立高斯样本的 SWD；回放基线 = 从训练集有放回抽 2000 条的 SWD。

## 实验

### E1 切断跨噪声注意力后，双时间步的收益还在吗？

- 层级 / 预算：L1；12 次训练，合计 ≤6 分钟。
- 运行位置：pc（CPU 即可；`/home/pc/lab/envs/main/bin/python`）；smoke 可在 mac。
- 旋钮：schedule ∈ {single, dual} × attention ∈ {full, sep}，共 4 格。固定：n=256，α=0.25，其余同共同设定。
- 朴素预期（Self-Flow 的解释）：dual+full 最好；切断后 dual+sep 退回 single+sep 的水平。
- 零信息基线 / 让效应消失的对照：三条参照线；single 两格就是“没有异质噪声”的对照。
- 指标：SWD（主）、mem（副）。
- 图规格：
  - `E1_main.png`：点图，x 为 4 格（固定顺序 single+full、dual+full、single+sep、dual+sep），y 为 SWD（对数轴）；每个 seed 一个小点、均值一个大点；三条参照线为水平线；在 dual+sep 处画灰色虚线标记朴素预期（= single+sep 的均值）。**支持增强假设**：dual+sep 明显低于虚线且与 dual+full 相当。**推翻**：dual+sep 落在虚线上而 dual+full 明显更低。**无法判定**：四格都落在彼此的 seed 范围内。
  - `E1_samples.png`：代表性 seed（按 dual+sep 的 SWD 取中位数）下，4 格各画前 12 条生成样本（不挑选，同一批噪声），每条样本画成 2D 闭合折线；再加一行 held-out 真实样本；各面板坐标轴相同。
- seeds：0 1 2。
- 验收：`results/E1.json` 含 4 格 × 3 seed 的 SWD、mem 和三条参照值。

### E2 这份收益像数据增强吗：数据越多越小？

- 层级 / 预算：L1；36 次训练，合计 ≤12 分钟。
- 运行位置：pc。
- 旋钮：n ∈ {64, 256, 1024, 4096}。在每个 n 上跑 3 个条件：single+full、dual+full、dual+sep。固定：α=0.25，步数 1500 不随 n 变。
- 朴素预期（交互/自监督解释）：收益与数据量无关，各 n 上 G 大致相同。
- 零信息基线 / 让效应消失的对照：三条参照线（回放基线随 n 变）；n=4096 是增强假设预言“效应消失”的条件。
- 指标：SWD；收益 G(n)=SWD(single+full)−SWD(dual 条件)，按 seed 配对；mem。
- 图规格：
  - `E2_main.png`：两个面板共享 x 轴（n，对数轴）。左：SWD（对数轴）对 n，三条线，用 `figstyle.seed_lines`，回放基线和真实下限画灰线。右：G(n) 对 n，两条线（dual+full、dual+sep），y=0 画细灰线；朴素预期“与数据量无关”画成灰色虚线，高度为 dual+full 在 n=64 处的 G 均值。**支持增强假设**：右图两条线随 n 单调降向 0。**推翻**：持平或上升。
  - `E2_mem.png`：mem 对 n，三条线；看 dual 是否在小 n 时更少背训练集。
- seeds：0 1 2。
- 验收：`results/E2.json` 含每个 n × 条件 × seed 的 SWD、mem、G。

### E3 AS 的代价：mask 比例与训练–采样不一致

- 层级 / 预算：L1；18 次训练，合计 ≤6 分钟。
- 运行位置：pc。
- 旋钮：α ∈ {0.25, 0.375, 0.5}（少数组 4 / 6 / 8 个 token）× ρ ∈ {0, 0.25}。ρ 是每个 batch 中改为“全部 token 同组、同一个 t、完整注意力”的样本比例。固定：schedule=dual，attention=sep，n=256。
- 朴素预期：两组总能覆盖整张图，比例无关紧要，三个 α 的 SWD 相同。
- 零信息基线 / 让效应消失的对照：三条参照线；ρ=0.25 是不一致假设预言“恶化消失”的条件。
- 指标：SWD、mem。
- 图规格：`E3_main.png`：折线图，x 为 α，y 为 SWD（对数轴），两条线（ρ=0、ρ=0.25），`figstyle.seed_lines`；灰色虚线为 α=0.25、ρ=0 的均值（朴素预期“持平”）。**支持不一致假设**：ρ=0 的线随 α 上升，ρ=0.25 的线基本持平。**推翻**：两条线都持平。
- seeds：0 1 2。
- 验收：`results/E3.json` 含 6 格 × 3 seed 的 SWD、mem。

## 运行

- 入口：`/home/pc/lab/envs/main/bin/python code/run.py --exp E1 --seeds 0 1 2`（在 `lab.py run start` 生成的运行目录中，于 `topics/2026-11-15-sra-selfflow-augmentation` 下运行）；`code/` 下一个 `run.py` 加一个 `toy.py`（数据、模型、训练、采样、指标），画图用 `tools/figstyle.py` 的 `setup / seed_lines / representative_seed / save_fig`。
- 登记：`tools/lab.py run start topics/2026-11-15-sra-selfflow-augmentation --exp E1 -- <上面的命令>`；结束后 `tools/lab.py run end <run-id> --status done|failed`。
- 输出：原始输出留在运行目录，指标写 `results/E*.json`，图写 `figures/E*_*.png`，拷回仓库提交。
- 顺序：先 smoke（`--exp E1 --seeds 0`，步数 100），确认单次 1500 步训练的耗时；若按耗时推算三个实验合计超过 30 分钟，停下报告，不自行改步数或模型。正式运行前把状态改为 running。
- 执行前告知：每次自动执行前先告诉用户机器 / 目录 / 命令 / 预计耗时与显存 / 读写哪些数据，确认后再执行。不需要 GPU，不下载数据或权重；依赖只有 torch、numpy、matplotlib，缺失时先说明再装，并把版本写进 SUMMARY。
- 停止条件：耗时超过预算 1.5 倍、出现 NaN 或发散、发现与假设无关的 bug、计划歧义会影响结论。遇到任何一种都停下，写进 SUMMARY 的执行记录并报告。不得为了得到预测的结果去调参、挑 seed 或改指标。

## 非目标

- 不复现论文的 FID/IS，不用 ImageNet、SiT 或任何预训练权重（L2/L3 不做）。
- 不检验“每步更多 t 样本”的竞争解释，不做多于两组的分组，不扫对齐损失的超参。
- toy 不能说明：真实图像上 DTS 的收益来源；有外部编码器（REPA）时的情形；大模型大数据下的结论。toy 的 token 只有 16 个且结构简单，“干净 token 引导嘈杂 token”在这里可能本来就不重要。

## 交付

- [ ] `results/E1.json`、`E2.json`、`E3.json`，各含 exp、status、seeds、metrics、representative_seed、runtime_sec
- [ ] `figures/` 中有 `E1_main.png`、`E1_samples.png`、`E2_main.png`、`E2_mem.png`、`E3_main.png`，且都实际打开看过
- [ ] SUMMARY.md 的每一节都已写满（含“这个 toy 不能说明什么”和偏离记录）
- [ ] `<python> tools/lab.py check topics/2026-11-15-sra-selfflow-augmentation --phase executed` 没有 ERROR

<!-- 批准后如需变更计划：在本文件末尾追加“## 计划修订”，写明日期、原因和改动；不要修改冻结预测。 -->
