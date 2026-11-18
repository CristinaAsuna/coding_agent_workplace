# 三机协作教程

你自己设计实验和方法，Claude 帮你审查并写成可执行的卡片，再让 agent 在合适的机器上执行，并留下能查的记录。本文教你怎么用：规则见 [AGENTS.md](AGENTS.md)，机器和数据在哪见 [MACHINES.md](MACHINES.md)。

---

## 1. 先理解三件事

**1）三台机器，各有分工。**

| | Mac | PC（WSL） | 集群 |
|---|---|---|---|
| 角色 | 指挥台：设计、写代码、分析 | 数据中心 + 单卡（实际约 6 GB 显存） | 大算力：Slurm、A100/V100 |
| 怎么进入 | 本机 | 在 Windows 上打开 Ubuntu 终端；在 Mac 上用 `ssh pc` | `ssh bme_login2` |
| 项目仓库 | `/Users/max/codebase/<项目>` | `~/lab/<项目>` | `/public_bme/home/maxquan/Projects/<项目>` |

Mac 不放数据、权重和缓存。显存 ≤6 GB 且几小时内能跑完的放 PC，其余放集群（`ssh pc`、`ssh windows`、`windows-linux` 的区别见 skill `lab-machines`，日常只用 `ssh pc`）。

**2）每个项目一个仓库，仓库就是它自己的共享记录。** 项目结构按 skill `research-project`：STATE.md 写当前主干和唯一的问题，每个实验一张卡片，批准后才写代码。三台机器各 clone 一份，`git pull` 就能看到别的机器做了什么。不属于任何项目的操作记在总台（本仓库）。

**3）`lab` 命令。** 三台机器上都装了，在任何目录下都能用；在项目仓库里它读写这个项目，不在仓库里就用总台。

| 命令 | 作用 |
|---|---|
| `lab status` | 先拉取最新提交，显示本机身份、正在进行的运行、最近的记录 |
| `lab run start <实验编号> -- <命令>` | 登记一次运行：快照已提交的代码，写运行记录，并推送（命令只记录，不执行） |
| `lab run end <run-id> --status done --outputs <位置>` | 运行结束，记下大产物放在哪 |
| `lab runs --active` | 还在跑的运行，以及所在目录和作业号 |
| `lab journal --days 3` | 最近 3 天各台机器上做过的事，带机器标签 |
| `lab log "做了什么"` | 零散操作记一笔（下载、装环境、清理）；`--hub` 强制记到总台 |
| `lab sync` | 会话结束时用：拉取其他机器的提交，再推送本机提交 |

另一台机器先 push 时，`lab` 会自动拉取对方的提交再推送；遇到冲突或有未提交的改动，它会停下来，保持原状。

---

## 2. 设计一个实验：你做设计，Claude 做审查

**第 1 步：你先写。** 在 Mac 的项目目录里，用自己的话写出：想验证什么、预测会看到什么数、什么结果会让你承认预测错了、打算怎么做。复现论文时，先弄清论文报告了哪些检查点，以及“论文想让你相信的那个数”是什么。

**第 2 步：让 Claude 审查**（它只提问，不替你换方案）：

> 我设计了一个实验：<你的设计>。用 research-project 的 §7 检查表审查它，只给问题清单，不要替我改方案。

它会检查：锚点是不是论文的最早检查点、效应是否先被确认、缩小规模有没有改掉机制依赖的条件、阈值有没有按噪声定、零结果有没有阳性对照、成本和止损点。你逐条回应、改自己的设计。

**第 3 步：写成卡片并批准。** 你满意以后：

> 按我现在的设计，用 tools/project.py new 开一张卡片，写好预测、推翻条件、预算和止损，给我批准。

批准后提交 `exp(<id>): card`，预测从此冻结。

---

## 3. 在目标机器上执行

按成本估计决定在哪台机器跑（Claude 会按 lab-machines 的路由给出建议，并说明理由）。

**在 PC 上**（Codex 或你自己，在 `~/lab/<项目>`）：

> git pull，按 experiments/<id>/CARD.md 在 pc 上执行。先告诉我机器、目录、命令、预计耗时和显存，我确认后再跑。用 `lab run start` 登记，在 tmux 里启动，并把 tmux 会话名作为 `--job`。

**在集群上**（登录节点只提交，不跑计算）：

> git pull，按 experiments/<id>/CARD.md 在集群上执行。按 $bme-cluster 先告诉我分区、GPU、时长和命令，我确认后再提交。用 `lab run start --job <作业号>` 登记，提交完就结束。

作业结束后，在同一台机器上：

> git pull，检查 run <run-id> 的状态（用 sacct 和日志确认，不要只看 squeue），把指标写进 result.json，`lab run end`，提交并 `lab sync`。

---

## 4. 看结果和状态

回到 Mac：

> lab status，再 lab journal --days 3，告诉我各机器最近做了什么、还有什么在跑。

读 `experiments/<id>/result.json` 和图，对照你写下的预测。结论和下一步写进卡片的“结果”一节，更新 STATE.md，提交并 `lab sync`。

**人在 Windows 或集群时**：在那台机器上同样先 `lab status`，它会先 pull，所以能看到别的机器的进度。

---

## 5. 零散小任务

在任何机器上都可以直接让 Codex 做，它会先告知再执行，做完用 `lab log` 记一笔：

| 想做的事 | 对 agent 说 |
|---|---|
| 查显存 | 用 $lab-machines，看 pc 现在空闲显存多少。 |
| 下载模型 | 用 $lab-machines，在 pc 上下载 <模型>，走 hf-mirror；下完登记到总台 MACHINES.md。 |
| 搬数据 | 用 $lab-machines，把 pc 上的 <数据集> 传到集群的 Datasets，走 bme_xfer。 |
| 清理 | 用 $lab-machines，列出 <机器> 上 runs/ 里超过 30 天的运行目录和大小，先不要删。 |

自己手动做了也要记：`lab log "手动下载 xxx 到 /public_bme/home/maxquan/Datasets/xxx"`。

---

## 6. 开场和收尾

**开场**（任何机器，先进入项目目录再启动 agent）：

> 先 git pull，再 lab status，告诉我本机是哪台、正在进行的运行是什么状态。

**收尾**：

> 把这次做的事记录好（运行用 lab run end，零散操作用 lab log），更新 STATE.md，commit 后 lab sync。

---

## 7. 出问题怎么办

| 现象 | 原因 | 处理 |
|---|---|---|
| push 被拒 | 另一台机器先 push 了 | `lab` 自动处理；提示有冲突或有未提交的改动时，先提交，再 `lab sync`，冲突让 agent 合并 |
| `ssh pc` 超时 | Windows 重启过，WSL 没有运行 | `ssh windows "wsl.exe -d Ubuntu-24.04 -u pc --exec true"` 唤醒，几秒后重试 |
| `lab run start` 拒绝 | 有未提交的代码 | 先提交（Markdown 文档的改动不拦） |
| 从 Mac 用 `ssh pc 'lab …'` 提示找不到 lab | 非交互 ssh 的 PATH 里没有 `~/.local/bin` | 写全路径 `ssh pc '~/.local/bin/lab status'`；在 WSL 终端里直接 `lab` 即可 |
| agent 没先问就执行 | 没在项目目录里启动，没读到规则 | 退出，进入项目目录后重开 |
| PC 显存不够 | Zotero 翻译模型占了约 6 GB | 改到集群；不要关掉翻译服务 |
| 集群作业从 squeue 消失 | 消失不代表成功 | 用 sacct 和日志确认 |
| 找不到某个数据或模型 | — | 先看 MACHINES.md 资产表，再 `lab journal` |
| 想在 Mac 上下载点东西 | — | 不要。让 PC 或集群下载，Mac 只通过 git 收发小文件 |

---

## 8. 为什么这样定

- **你来设计**：自己提出问题、预测和推翻条件，才能练出判断；Claude 的价值在于挑出你没想到的可行性问题，而不是替你想。
- **锚点和先确认效应**：此前缩小到 toy 的复现，改掉了机制依赖的条件，花了资源却得不到可判断的结论。以论文最早的检查点为锚，先看到效应，再做归因。
- **登记和告知**：三台机器上都有 agent，不记录就搞不清谁在哪做了什么；先告知再确认，挡住跑错机器、写错目录、占满显存。
