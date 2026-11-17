# From SRA to Self-Flow: Data Augmentation or Self-Supervision?（MNIST L2）：实验计划

topic：2026-11-16-sra-selfflow-mnist · 发现过程重建：[DISCOVERY.md](DISCOVERY.md) · 批准日期：2026-11-16（含 pc 10 GPU 小时的预算例外）

## 目标

在 MNIST 上用一个小 DiT 亲历论文的核心实验：(E1) 双时间步是否带来收益，切断跨噪声注意力后收益是否保留；(E2) 直接测量训练时“干净 token 帮助嘈杂 token”的交互是否真的发生。前一个合成 toy（topics/2026-11-15-sra-selfflow-augmentation）里双时间步没有可辨认的收益，本 topic 换成有空间冗余的真实图像。执行者只需本文件。

## 冻结预测

| 实验 | 用户预测 | agent 预测 | 推翻条件（针对 agent 预测） |
|---|---|---|---|
| E1a 双时间步收益（2 万步，dual+full 对 single+full） | “明显更好” | 差不多：两者 FID 均值之差小于两条件各自 seed 范围（max−min）中较大的那个 | 均值之差大于该范围（无论哪个方向） |
| E1b 切断注意力（2 万步，dual+sep 对 dual+full） | “相当” | 更差：dual+sep 的 FID 均值更高，且差值大于 seed 范围；但不超过 dual+full 的 1.5 倍 | 差值小于 seed 范围，或 dual+sep 更低 |
| E2 交互是否发生（dual+full 模型，上下文从 t=0.8 变到 t=0） | “大幅下降” | 嘈杂 token 上的速度 MSE 下降至少 50%；single+full 模型也下降但比例更小 | dual+full 模型下降不到 30% |

“seed 范围”一律指被比较的两个条件各自 3 个 seed 的 max−min 中较大的那个。

## 共同设定（除旋钮外不得改动）

- **数据**：MNIST，训练集 60000 张，28×28，像素缩放到 [−1, 1]，类别条件 10 类。用 torchvision 下载到 `pc:~/lab/assets/datasets/mnist/`（约 11 MB），下载后登记到 MACHINES.md 资产表。测试集 10000 张只用于评估。
- **模型**：DiT，patch 2 → 14×14=196 个 token（每个 4 维），宽 192，深 8，6 头，MLP 比例 4，固定 2D 正弦位置嵌入。条件向量 = 逐 token 的时间步嵌入 + 类别嵌入，经逐 token adaLN（每个 block 输出 shift/scale/gate，gate 零初始化）注入；输出层零初始化，输出 4 维速度。类别 dropout 0.1；采样不用 CFG。
- **flow matching**：`x_t=(1−t)x0+t·x1`，x1~N(0,I)，目标 `x1−x0`，逐 token MSE 等权平均；t~U(0,1)。
- **分组**：每张图随机取恰好 ⌊α·196⌋ 个 token 为第 0 组（少数组），α=0.25 即 49 / 147。
  - schedule=single：两组共用一个 t；dual：两组各自独立采样 t，共用同一份噪声。
  - attention=full：不加掩码；sep：所有 block 只允许同组 token 互相注意（`A_ij=[g_i==g_j]`）。single+sep 也照样分组。
- **SRA 对齐损失（固定开启）**：EMA 教师，decay 0.999；学生第 3 个 block 之后的特征经 head（Linear 192→384、SiLU、Linear 384→192）与教师第 6 个 block 之后的特征做逐 token `1−cos`（沿通道维），教师无梯度；教师输入用与学生相同的 x0、x1、类别、分组和注意力模式，各组时间步为学生的 t 减 U(0,0.2) 后截断到 ≥0；总损失 = 速度损失 + 0.5·对齐损失。
- **训练**：AdamW，lr 3e-4，无 weight decay，batch 256，20000 步，fp16 自动混合精度，单卡。在 5000、10000、20000 步保存 EMA 权重并评估。
- **采样**：EMA 权重，Euler 50 步从 t=1 到 0，全部 token 同一个 t、完整注意力；每类 1000 张共 10000 张，噪声 seed 固定且各条件相同。
- **评估分类器**：一个小 CNN（两层卷积 + 全连接，倒数第二层 64 维），用 seed 0 在 MNIST 训练集上训 2 个 epoch，全部运行共用同一份权重；测试准确率低于 98.5% 就停下报告。
- **主指标 FID（越低越好）**：分类器倒数第二层特征上的 Fréchet 距离，10000 张生成图对 10000 张测试图。
- **副指标 cond-acc**：生成图被分类器判为其条件类别的比例。
- **参照**：真实下限 = 训练集随机 10000 张对测试集的 FID；零信息基线 = 逐像素匹配均值方差的独立高斯图像的 FID。

## 实验

### E1 真实图像上的 2×2：收益在不在，切断后还在不在？

- 层级 / 预算：L2；12 次训练（4 条件 × 3 seed），预计每次约 40 分钟，合计约 8 GPU 小时（未实测，以 smoke 为准）。
- 运行位置：pc（RTX 3060，预计显存 <3 GB；运行前用 `nvidia-smi` 确认空闲 ≥4 GB）。用 tmux。
- 旋钮：schedule ∈ {single, dual} × attention ∈ {full, sep}。固定：α=0.25，其余同共同设定。
- 朴素预期（Self-Flow 的解释）：dual+full 最好；切断后 dual+sep 退回 single+sep 的水平。
- 零信息基线 / 让效应消失的对照：两条参照线；single 两格是“没有异质噪声”的对照。
- 指标：5000 / 10000 / 20000 步的 FID（主）和 cond-acc（副）。冻结预测只针对 20000 步。
- 图规格：
  - `E1_curve.png`：折线图，x 为训练步数（5000、10000、20000），y 为 FID（对数轴），4 条线，每个 seed 一条细线、均值一条粗线；真实下限画水平灰线。
  - `E1_final.png`：点图，x 为 4 格（固定顺序 single+full、dual+full、single+sep、dual+sep），y 为 20000 步的 FID；小点是各 seed，大点是均值；在 dual+sep 处画灰色虚线标记朴素预期（= single+sep 的均值）。**支持增强假设**：dual 两格都明显低于各自的 single，且 dual+sep 与 dual+full 相当。**支持交互解释**：dual+full 明显低于 single+full，而 dual+sep 落在虚线上。**无法判定**：四格都在彼此的 seed 范围内。
  - `E1_samples.png`：代表性 seed（按 dual+sep 的 20000 步 FID 取中位数）下，4 个条件各一个面板，每类前 2 张共 20 张，不挑选，各条件同一批噪声。
- seeds：0 1 2。
- 验收：`results/E1.json` 含 4 条件 × 3 seed × 3 个检查点的 FID、cond-acc 和两个参照值。

### E2 交互在训练时真的发生了吗？

- 层级 / 预算：L2；不训练，只对 E1 的 20000 步 EMA 权重做前向，合计 ≤10 分钟。
- 运行位置：pc。
- 做法：取测试集前 2000 张，每张用固定 seed 生成一个 49 / 147 的分组和一份噪声。少数组固定在 t=0.8；多数组（上下文）的时间步 t_ctx ∈ {0.8, 0.6, 0.4, 0.2, 0.0}。测少数组 token 上的速度 MSE。
- 旋钮：t_ctx。四条线：
  1. dual+full 模型，完整注意力（主线）；
  2. single+full 模型，完整注意力（训练时没见过异质噪声）；
  3. dual+sep 模型，按分组切断注意力（对照：少数组看不到上下文）；
  4. dual+sep 模型，完整注意力（它采样时的工作方式）。
- 朴素预期（“交互没被用上”）：四条线都是水平线。
- 零信息基线 / 让效应消失的对照：第 3 条线按构造必须完全水平，同时充当实现正确性检查；若它随 t_ctx 变化超过 1%，说明掩码有 bug，停下报告。
- 指标：少数组速度 MSE；下降比例 = 1 − MSE(t_ctx=0) / MSE(t_ctx=0.8)。
- 图规格：`E2_main.png`：折线图，x 为 t_ctx（从 0.8 到 0，左嘈杂右干净），y 为少数组速度 MSE，4 条线，seed 细线加均值粗线；灰色虚线为主线在 t_ctx=0.8 处的均值（朴素预期“持平”）。**支持“交互发生”**：主线明显下降而第 3 条线水平。**推翻**：主线基本水平。
- seeds：0 1 2（对应 E1 的三个 seed 的权重）。
- 验收：`results/E2.json` 含 4 条线 × 5 个 t_ctx × 3 seed 的 MSE 和下降比例。

## 运行

- 入口：`/home/pc/lab/envs/main/bin/python code/run.py --exp E1 --seeds 0 1 2`（在 `lab.py run start` 生成的运行目录中，于 `topics/2026-11-16-sra-selfflow-mnist` 下运行）。
- 登记：`tools/lab.py run start topics/2026-11-16-sra-selfflow-mnist --exp E1 -- <上面的命令>`；结束后 `tools/lab.py run end <run-id> --status done|failed --outputs <权重位置>`。
- 输出：权重和样本留在 pc 的运行目录（E2 要读），指标写 `results/E*.json`，图写 `figures/E*_*.png`，拷回仓库提交。
- 预算：本 topic 的预算是 pc 上 10 GPU 小时，这是用户批准本计划时给出的例外，不修改 lab.toml。
- 顺序：先 smoke（single+full 和 dual+sep，seed 0，各 200 步，加一次 1000 张的采样与 FID），确认管线并测速；若按测速推算 E1 合计超过 10 GPU 小时，停下报告，不自行改步数或模型。正式运行前把状态改为 running。
- 执行前告知：每次自动执行（含下载 MNIST）前先告诉用户机器 / 目录 / 命令 / 预计耗时与显存 / 读写哪些数据，确认后再执行。依赖只有 torch、torchvision、numpy、scipy、matplotlib，缺失时先说明再装，并把版本写进 SUMMARY。
- 停止条件：耗时超过预算 1.5 倍、出现 NaN 或发散、发现与假设无关的 bug、计划歧义会影响结论。遇到任何一种都停下，写进 SUMMARY 的执行记录并报告。不得为了得到预测的结果去调参、挑 seed 或改指标。

## 非目标

- 不复现论文的 FID/IS 数字，不用 ImageNet、SiT 或预训练权重；不扫 mask 比例（前一个 topic 的 E3 已做），不扫对齐损失超参。
- 这个实验不能说明：ImageNet 尺度、潜空间、大模型下的结论；这里的 FID 用自训分类器特征，数值不能与任何论文比较。MNIST 背景大片为空，token 间冗余的形态与自然图像不同。

## 交付

- [ ] `results/E1.json`、`E2.json`，各含 exp、status、seeds、metrics、representative_seed、runtime_sec
- [ ] `figures/` 中有 `E1_curve.png`、`E1_final.png`、`E1_samples.png`、`E2_main.png`，且都实际打开看过
- [ ] SUMMARY.md 的每一节都已写满（含“不能说明什么”和偏离记录）
- [ ] MNIST 已登记到 MACHINES.md 资产表
- [ ] `/home/pc/lab/envs/main/bin/python tools/lab.py check topics/2026-11-16-sra-selfflow-mnist --phase executed` 没有 ERROR

<!-- 批准后如需变更计划：在本文件末尾追加“## 计划修订”，写明日期、原因和改动；不要修改冻结预测。 -->

## 计划修订

### 2026-11-16：改为单个 seed（用户要求）

- 原因：用户明确表示“不需要使用多个seed”。本修订在任何实验运行之前写入。
- 改动：E1 和 E2 只用 seed 0。E1 为 4 次训练（预计合计约 2.7 GPU 小时，未实测）；E2 只用这 4 份权重。预算上限相应改为 4 GPU 小时，smoke 推算超过即停下报告。凡正文写 `--seeds 0 1 2` 处改为 `--seeds 0`，`representative_seed` 记 0。
- 判定标准的替换：冻结预测里的“seed 范围”在单 seed 下没有定义，统一替换为固定阈值：两个条件在 20000 步的 FID 相对差 `|a−b| / min(a,b)` 超过 10% 记为“明显不同”，否则记为“相当 / 差不多”。预测本身的文字不变。E1b 的“不超过 1.5 倍”和 E2 的阈值不受影响。
- 图：不再画 seed 细线，每个条件只有一个点或一条线。
- 代价（必须写进 SUMMARY 的“不能说明什么”）：单 seed 无法估计训练随机性；10% 阈值是事先指定、未经校准的，落在阈值附近的差异不能下结论。论文自身也是单 seed。

### 2026-11-16：预算改为 10 GPU 小时（用户批准）

- 原因：smoke 按较慢条件测速外推约 8.31 小时，超过此前的 4 小时预算；用户同意提高至 10 小时。
- 改动：本 topic 的计算预算上限为 10 GPU 小时，包含已用的 smoke 时间。其余实验设定与冻结预测不变。
<!-- approved-gpu-budget-seconds: 36000 -->

### 2026-11-16：修复 FP16 溢出处理

- 原因：E1 的 `single+full` 在约 3100 步后因 Inf 梯度停止；独立重放发现高 loss scale 下溢出，同 batch 在 FP16 scale 1 与 FP32 下梯度有限。用户要求考虑并修复。
- 改动：前向非有限和梯度 NaN 仍停止；Inf 梯度由 GradScaler 跳过该次更新并降低 scale，跳步时不更新 EMA；scale 已不大于 1 仍出现 Inf 则停止。记录跳步原因、scale 和成功更新数；20000 步仍指迭代次数。不得改变模型、学习率或冻结预测。
- 验证通过后，正式续跑需另行确认；新运行从原运行的 3000 步完整检查点恢复，沿用原单 seed 并计入此前 smoke、正式运行和诊断时间。原运行快照和检查点保留。
