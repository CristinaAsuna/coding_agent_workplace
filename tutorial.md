# 三机科研工作流教程

这份教程教你同时用好 Mac、Windows PC（WSL）和 BME 集群三台机器：在哪台机器上做什么、对 agent 说什么、怎么知道每台机器上发生了什么。规则细节见 [AGENTS.md](AGENTS.md) 和 [MACHINES.md](MACHINES.md)，本文只讲怎么用。

---

## 1. 先理解三件事

**1）一个仓库，三份克隆。** GitHub 上的 `CristinaAsuna/coding_agent_workplace` 在三台机器上各有一份。方案、结果、运行记录、零散操作的记录都放在里面，`git pull` 一下就能看到别的机器做了什么。

| | Mac | PC（WSL） | 集群 |
|---|---|---|---|
| 角色 | 指挥台：写方案、分析、复盘 | 数据中心 + 单卡（实际约 6 GB 显存） | 大算力：Slurm、A100/V100 |
| 怎么进入 | 本机 | 在 Windows 上打开 Ubuntu 终端；在 Mac 上用 `ssh pc` | `ssh bme_login2` |
| 仓库路径 | `/Users/max/codebase/coding_agent_repro` | `~/lab/coding_agent_repro` | `/public_bme/home/maxquan/Projects/coding_agent_repro` |
| 运行 lab.py 的 Python | `/Users/max/codebase/.ml/.venv/bin/python` | `~/lab/envs/main/bin/python` | `/public_bme/home/maxquan/envs/uv-python/cpython-3.12-linux-x86_64-gnu/bin/python3.12` |

**2）agent 自动读规则，前提是在仓库目录里启动。** Claude 和 Codex 启动时会读两层规则：
- **全局说明**：告诉 agent 本机是哪台、仓库在哪、要先告知再执行。Mac 上的 Claude 读 `~/.claude/CLAUDE.md`，三台机器上的 Codex 读 `~/.codex/AGENTS.md`。
- **仓库规则**：仓库根目录的 `AGENTS.md`，Claude 通过 `CLAUDE.md` 导入它。

两个 skill 在三台机器上都已安装：
- `research-discovery-lab`：科研流程，包括开题、执行、复盘。
- `lab-machines`：三机协作。

在 Claude 里直接说“用 research-discovery-lab …”；在 Codex 里写 `$research-discovery-lab`、`$lab-machines`。

**3）四个习惯。**

1. 开始先 `git pull`，结束要 push。
2. agent 执行任何操作前都会先告诉你“机器 / 目录 / 命令 / 预计耗时与显存 / 读写哪些数据”。你确认了它才执行。
3. 正式实验用 `lab.py run` 登记；零散操作用 `lab.py log` 记一笔。
4. Mac 上不下载数据，也不放模型权重。

---

## 2. 每次会话的开场和收尾

在任何机器上启动 agent 之前，先进入那台机器的仓库目录。

**开场**（第一句话就这么说）：

> 先 git pull，再运行 lab.py status，告诉我本机是哪台、各 topic 和正在进行的运行是什么状态。

**收尾**：

> 把这次做的事按 AGENTS.md 记录好（运行用 lab.py run end，零散操作用 lab.py log），commit 并 push。

---

## 3. 完整走一遍：以 Grokking 为例

下面假设这个 topic 有两个实验：E1 是小模型复现“过拟合很久之后突然泛化”，放在 PC 上跑；E2 是扫多组权重衰减和数据比例，计算量大，放在集群上跑。具体分配以 PLAN 为准。

### 第 1 步：在 Mac 上开题（Claude）

用 Claude 桌面版的话，把工作目录选成 `coding_agent_repro`；用命令行的话：

```bash
cd /Users/max/codebase/coding_agent_repro && claude
```

建议切到 **Plan mode**，然后说：

> 先 git pull 并运行 lab.py status。然后用 research-discovery-lab 开一个 topic：Grokking（Power et al., 2022, arXiv:2201.02177）。

接下来会发生这些事：

1. Claude 先问你 2–3 个预测问题，比如“你觉得作者最先看到了什么现象”“训练到第几步测试准确率会突然上升”。认真回答，这是练习的核心。回答之后才会看到它对发现过程的重建。
2. 它给出计划（DISCOVERY + PLAN）。你重点看三处：
   - 冻结预测：你的回答是否原样记录了。
   - 每个实验的**运行位置**：mac / pc / cluster，以及选择的理由。
   - 实验分在多台机器时，由**哪台机器负责最后汇总 SUMMARY.md**。默认是跑最后一个实验的那台；其他机器只提交自己的结果，不改 SUMMARY 的其他部分。
3. 你批准后，它会建好 `topics/2026-xx-xx-grokking/`，提交 `plan[mac](<id>)` 并 push。从这一刻起，你的预测冻结，不能再改。

如果你想改分配（比如把 E2 也放到 PC 上），在批准前直接告诉它。

### 第 2 步：在 PC 上执行 E1（Codex）

**进入 PC**，两种方式任选：

- 人在 Windows 前：打开 Ubuntu 终端。
- 人在 Mac 前：在 PC 上开一个 tmux 会话，在里面运行 Codex。这样 Mac 合盖或断网，Codex 也不会中断：

  ```bash
  ssh -t pc tmux new -A -s codex
  ```

**启动 Codex：**

```bash
cd ~/lab/coding_agent_repro && codex
```

**对 Codex 说：**

> git pull，然后按 AGENTS.md 执行 topics/<id>/PLAN.md 中运行位置为 pc 的实验（E1）。

**过程中你要做的：** 确认它报上来的执行计划。比如它说“在 pc 的 ~/lab/runs/… 下用 tmux 运行 python code/run.py，预计 10 分钟、显存 2 GB”，看过没问题再同意。它会依次：

1. 写代码，提交；
2. 先跑一个小规模的 smoke 测试；
3. 用 `lab.py run start` 登记，在 tmux 里正式运行；
4. 跑完后把小结果和图拷回 topic，执行 `lab.py run end`，然后 push。

**中途想看进度**（在 PC 上）：

```bash
tmux ls
```

```bash
cd ~/lab/coding_agent_repro && ~/lab/envs/main/bin/python tools/lab.py runs --active
```

**如果 tmux 会话断了：** 再执行一次 `ssh -t pc tmux new -A -s codex` 就能回到原来的 Codex。训练是在另一个 tmux 会话里跑的，不受影响。

### 第 3 步：在集群上执行 E2（Codex）

```bash
ssh bme_login2
```

```bash
cd /public_bme/home/maxquan/Projects/coding_agent_repro && codex
```

**第一次对话：提交作业**

> git pull，然后按 AGENTS.md 执行 topics/<id>/PLAN.md 中运行位置为 cluster 的实验（E2），Slurm 相关按 $bme-cluster 的规则。提交完作业就告诉我作业号，然后结束。

**为什么要分两次对话：** 集群登录节点上的进程运行超过约 1 小时会被杀掉，Codex 也是登录节点上的一个进程，不能一直开着等作业跑完。所以第一次对话只负责提交作业，作业号会通过 `--job` 记进运行记录。

**作业结束后，第二次对话：收尾**

> git pull，检查 run <run-id> 的作业状态（sacct），把小结果和图拷回 topic，按 PLAN 汇总 SUMMARY.md，然后 lab.py run end 并 push。

忘了 run-id 时，在 Mac 或任何机器上查：

```bash
cd /Users/max/codebase/coding_agent_repro && git pull -q && /Users/max/codebase/.ml/.venv/bin/python tools/lab.py runs --active
```

### 第 4 步：回 Mac 复盘（Claude，开一个新会话）

最好开一个新会话，不要沿用写方案的那个，这样复盘时不会带着当初的预设：

> git pull，然后用 research-discovery-lab review topics/<id>。

Claude 会：
1. 核对 SUMMARY 中的数字和结果文件是否一致，逐张看图。需要看大产物时，它会按 lab-machines 的规则只读登录 `ssh pc` 或 `ssh bme_login2` 查看。
2. 和你一起复盘：逐个对照你当初的预测，再请你自己写 taste 判断。这部分它不会代写。
3. 更新 EXPERIMENTS、INSIGHTS、LEARNING 等根目录文件，提交 `review[mac](<id>)` 并 push。

---

## 4. 人在 Windows 前或在集群上：查看当前情况

在任何一台机器上，进入仓库目录，先执行 `git pull`，再用该机器的 Python（路径见第 1 节的表）运行：

| 想知道 | 命令（`<py>` 换成该机器的 Python） |
|---|---|
| 本机是哪台、各 topic 进行到哪一步、哪台机器上在跑什么 | `<py> tools/lab.py status` |
| 最近几天三台机器上都做了什么（按时间排列，带机器标签） | `<py> tools/lab.py journal --days 3` |
| 只看某台机器 | `<py> tools/lab.py journal --days 7 --machine cluster` |
| 还在跑的运行，以及它们的目录和作业号 | `<py> tools/lab.py runs --active` |
| 某个数据或模型放在哪里 | 打开 `MACHINES.md` 的资产表 |

在 PC 上，一行就能看最近 3 天的记录：

```bash
cd ~/lab/coding_agent_repro && git pull -q && ~/lab/envs/main/bin/python tools/lab.py journal --days 3
```

也可以直接问 agent：

> git pull，用 lab.py journal 和 runs 告诉我最近三天每台机器上做了什么、还有什么在跑。

**人在 Windows 前时**，一律用 Ubuntu 终端（WSL）里的 Codex。Windows 原生 PowerShell 里的 Codex 没有这套全局说明，也没有仓库克隆，它做的事不会被记录。

---

## 5. 零散小任务：任何机器、任何 Codex

比如下载模型、装依赖、看集群排队、清理磁盘。同样要**先进入仓库目录再启动 Codex**，然后说任务，并提一句 `$lab-machines`：

| 场景 | 可以这样说 |
|---|---|
| 下载模型 | 用 $lab-machines，把 facebook/dinov2-large 下载到集群的 Datasets 下，走 hf-mirror。 |
| 在 PC 上装依赖 | 用 $lab-machines，在 pc 的 ~/lab/envs/main 里装 timm，并记录版本。 |
| 看集群资源 | 用 $lab-machines，看一下集群现在哪个分区最快能拿到一张 A100，只读，不提交。 |
| 搬数据 | 用 $lab-machines，把 pc 上 ~/lab/assets/datasets/aariz 传到集群的 Datasets，走 bme_xfer。 |
| 清理 | 用 $lab-machines，列出 pc 上 ~/lab/runs 里超过 30 天的运行目录和大小，先不要删。 |

Codex 会先告知再执行；做完后把新数据或模型登记进 `MACHINES.md`，用 `lab.py log` 记一笔，然后 push。

就算是只花两分钟的小事也要记，比如你自己在集群上手动下载了一个文件。可以自己补一条：

```bash
cd /public_bme/home/maxquan/Projects/coding_agent_repro && /public_bme/home/maxquan/envs/uv-python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 tools/lab.py log "手动下载 xxx 到 /public_bme/home/maxquan/Datasets/xxx"
```

科研流程的 skill 在每台机器上都能用。比如人在 Windows 前时，也可以在 PC 的 Codex 里说“`$research-discovery-lab` review topics/<id>”，不一定要回到 Mac。

---

## 6. 提示词速查

| 用途 | 在哪 | 提示词 |
|---|---|---|
| 开场 | 任何机器 | 先 git pull，再运行 lab.py status，告诉我本机是哪台、各 topic 和正在进行的运行是什么状态。 |
| 开题 | Mac / Claude | 用 research-discovery-lab 开一个 topic：<论文>。 |
| 执行 | PC / Codex | git pull，然后按 AGENTS.md 执行 topics/<id>/PLAN.md 中运行位置为 pc 的实验。 |
| 提交作业 | 集群 / Codex | git pull，然后按 AGENTS.md 执行 topics/<id>/PLAN.md 中运行位置为 cluster 的实验，按 $bme-cluster 提交，交完就结束。 |
| 作业收尾 | 集群 / Codex | git pull，检查 run <run-id> 的状态，拷回结果，按 AGENTS.md 收尾并 push。 |
| 复盘 | Mac / Claude（新会话） | git pull，然后用 research-discovery-lab review topics/<id>。 |
| 查进度 | 任何机器 | git pull，用 lab.py journal 和 runs 告诉我最近每台机器上做了什么、还有什么在跑。 |
| 零散任务 | 任何机器 / Codex | 用 $lab-machines，<任务>。 |
| 收尾 | 任何机器 | 把这次做的事按 AGENTS.md 记录好，commit 并 push。 |
| 定期回顾 | Mac / Claude | 用 research-discovery-lab 做一次 retro。（每复盘完 5 个 topic 做一次） |

---

## 7. 出问题怎么办

| 现象 | 原因 | 处理 |
|---|---|---|
| push 被拒（rejected / non-fast-forward） | 另一台机器先 push 了 | `git pull --rebase && git push`。lab.py 推送失败时也只是提示，不会丢失记录 |
| `ssh pc` 超时 | Windows 重启过，WSL 没有运行 | `ssh windows "wsl.exe -d Ubuntu-24.04 --exec true"` 唤醒 WSL，几秒后重试 |
| agent 没先问就开始执行，或说不清本机是哪台 | 没在仓库目录里启动 | 退出，进入仓库目录后重开 |
| PC 显存不够 | Zotero 翻译模型占了约 6 GB | 让 agent 把这个实验改到集群上跑；不要擅自关掉翻译服务 |
| 集群作业从 squeue 里消失了 | 消失不代表成功 | 让 Codex 用 sacct 和日志确认作业状态，再做收尾 |
| 两台机器都改了 SUMMARY.md，产生冲突 | 一个 topic 只能由一台机器汇总 SUMMARY | 在 PLAN 里指定汇总者；冲突时让 agent 合并，两边内容都保留 |
| 找不到某个数据或模型 | — | 先看 `MACHINES.md` 资产表，再用 `lab.py journal` 搜关键词 |
| 想在 Mac 上下载点东西试试 | — | 不要。让 PC 或集群下载，Mac 只通过 git 收发小文件 |

---

## 8. 这些规则为什么这样定

- **为什么要登记运行和记录零散操作？** 你之前的困扰就是记不清在哪台机器上做了什么。现在每个动作都会变成一个带机器标签的提交，存在三台机器共享的仓库里，`journal` 一查就清楚。
- **为什么执行前要先告知？** 三台机器上都有 Codex，你很容易搞不清某条命令是在哪台机器上跑的。先告知、再确认，能挡住跑错机器、写错目录、占满显存这类问题。
- **为什么 Mac 不放数据？** Mac 的硬盘小，而且最常合盖。数据放在 PC 和集群上，每台机器用到什么就自己下载；机器之间的大文件走 PC↔集群的传输节点，不经过 Mac。
- **为什么复盘要开新会话？** 写方案的会话会带着当初的预设，换一个干净的会话来核对结果更客观，也更容易发现问题。
