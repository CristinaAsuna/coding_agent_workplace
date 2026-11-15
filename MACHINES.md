# 机器与数据

本文件回答三个问题：每台机器做什么、东西放在哪、某个数据或权重在哪里。

- 运行记录：`tools/lab.py runs`
- 各机器做过什么：`tools/lab.py journal`
- 机器配置：lab.toml `[machines.*]`
- 方法细节：skill `lab-machines`

## 机器

| 标识 | 角色 | 连接（从 Mac） | 仓库路径 | 跑什么 |
|---|---|---|---|---|
| mac | 指挥台 | 本机 | `/Users/max/codebase/coding_agent_repro` | 规划、写代码、分析；只跑 L0 和 ≤5 分钟的 CPU smoke |
| pc | 数据中心 + RTX 3060（实际可用约 6 GB） | `ssh pc`（WSL，2222 端口） | `~/lab/coding_agent_repro` | L1/L2，用 tmux 运行 |
| cluster | 大算力（Slurm） | `ssh bme_login2`；传输走 `bme_xfer` | `/public_bme/home/maxquan/Projects/coding_agent_repro` | L3，遵循 bme-cluster |

## 路由规则

1. 无 GPU 且 ≤5 分钟 → mac。
2. 单卡、显存 ≤6 GB、耗时 ≤数小时 → pc。运行前用 `nvidia-smi` 确认空闲显存，Zotero 翻译模型会占用一部分。
3. 更大的显存、多卡或长时间训练 → cluster。
4. 数据已经在哪台机器上，优先在那台跑。

## 路径约定

| 机器 | 运行目录 | 热数据与缓存 | 权重 | 冷归档 |
|---|---|---|---|---|
| mac | `topics/<id>/runs/`（只放 smoke） | 不放 | 不放 | — |
| pc | `~/lab/runs` | `~/lab/assets`（ext4，读写快） | `/mnt/d/vscode_workplace/brainstorm/models`（HF 缓存在其下的 `huggingface/`） | `/mnt/d/lab-archive` |
| cluster | `/public_bme/home/maxquan/Projects/runs` | `/public_bme/home/maxquan/Datasets` | 各项目自行管理，或放在 Datasets 下 | — |

- Python 环境：
  - pc：`~/lab/envs/main`（uv，torch CUDA）。
  - cluster：tools/lab.py 用 uv 安装的 Python 3.12；实验环境按 topic 用 conda 或 uv 另建。
- 下载：pc 和 cluster 都设置 `HF_ENDPOINT=https://hf-mirror.com`，只对当次命令生效。

## 传输方式

- PC ↔ 集群：在 WSL 上 `rsync -aH --partial <源>/ bme_xfer:<目标>/`，不经过 Mac。
- Mac → PC：只用于迁移旧数据，用 `/opt/homebrew/bin/rsync`。
- 小结果和图：一律通过 git。

## 资产表

新下载或迁移的数据和权重都要登记在这里。

| 名称 | 来源 / revision | 位置（机器:路径） | 大小 | 使用的 topic |
|---|---|---|---|---|
| PC 现有权重库（dit、sit、huggingface、torchhub） | 历史下载 | pc:`/mnt/d/vscode_workplace/brainstorm/models` | 18 GB | — |
