# coding_agent_repro：三机总台

Mac（指挥台）、Windows PC 的 WSL（数据中心 + RTX 3060）和 BME 集群（大算力）三台机器共用的一个“总台”仓库：

| 文件 | 作用 |
| --- | --- |
| [tutorial.md](tutorial.md) | **怎么用**：设计实验、让 agent 在三台机器上执行、查看谁在哪做了什么 |
| [MACHINES.md](MACHINES.md) | 每台机器做什么、东西放在哪、数据和权重的资产表 |
| [AGENTS.md](AGENTS.md) · [CLAUDE.md](CLAUDE.md) | 给 agent 的规则（Claude 通过 CLAUDE.md 导入） |

实验不在这里做，每个研究项目有自己的仓库（按 skill `research-project` 搭建）；多机记录用 `lab` 命令（skill `lab-machines`）。此前的“发现过程复现”工作区（topics、规则、成长档案）已在 2026-11-18 下线，内容仍在 git 历史里。
