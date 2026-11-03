# 科研发现复现实验室：永久规则

本工作区用来复现论文作者的发现过程（观察 → 推理 → 实验），用小而快的 toy 实验和可视化亲历它，借此培养用户的科研能力和 taste。用户是初级研究者，**用户的成长是首要目标，实验结果只是手段**。默认用中文，代码和文件名用英文。

规则末尾的 `[H-xxx]` 指向 HARNESS_LOG.md 中的来源；`[H-000]` 表示初始规则。

## 0. 新会话先做

1. 运行 `/Users/max/codebase/.ml/.venv/bin/python tools/lab.py status`，再读 EXPERIMENTS.md 和 LEARNING.md。LEARNING.md 里的当前阶段决定你提什么问题、什么时候揭晓答案。
2. 只读与本次任务相关的 topic 文件；规划新 topic 时再读 INSIGHTS.md 和 RESEARCH_MOVES.md。不要读 runs/ 和大文件。
3. 确定自己的角色（Planner / Executor / Reviewer），按下面对应章节工作。用户只想讨论时直接讨论，不建 topic。
4. 方法细节在 skill `research-discovery-lab` 中；未安装时直接读 `/Users/max/codebase/research_plugins/Research_skills/research-discovery-lab/references/`。[H-000]

## 1. Topic 生命周期 [H-000]

状态记录在 EXPERIMENTS.md 中：`discovery → planned → running → executed → reviewed`，另有 `blocked` 和 `abandoned`。

| 状态 | 负责 | 进入条件 |
| --- | --- | --- |
| discovery | Planner | 已用 `tools/lab.py new` 建好 topic |
| planned | Planner | DISCOVERY 和 PLAN 已填完，用户批准，提交 `plan(<id>): …` 冻结预测 |
| running | Executor | smoke 已通过，正式运行开始 |
| executed | Executor | `check --phase executed` 无 ERROR，提交 `run(<id>): …` |
| reviewed | Reviewer | 已和用户完成复盘，根目录文件已更新，`check --phase reviewed` 无 ERROR，提交 `review(<id>): …` |

- 新建 topic 只用 `/Users/max/codebase/.ml/.venv/bin/python tools/lab.py new <slug> --paper "<标题或 arXiv>"`，不要手工建目录。
- 推进状态前先运行 `tools/lab.py check topics/<id>`；有 ERROR 就不推进。
- 只做本地提交，不推送，不建远程仓库。

## 2. 文件纪律（防臃肿）[H-000]

- 每个 topic 只有四个 MD：DISCOVERY.md、PLAN.md、SUMMARY.md、REFLECTION.md。不新增 HANDOFF、PROGRESS、NOTES 之类的文件；需要新的文件类型时先走 harness 提案。
- topic 目录结构：
  - `code/`：入口为 `code/run.py --exp E1 --seeds 0 1 2`。
  - `figures/`：SUMMARY 引用的正式图，进 git。
  - `results/`：`E*.json` 小指标，进 git。
  - `runs/`：原始输出、日志、checkpoint、论文缓存，不进 git。
- 各文件的行数上限写在 lab.toml；超限时合并、删减，不另开新文件。
- 根目录的总结文件只做合并式更新：改写已有条目，不按时间往下堆。

## 3. Planner

1. 读论文原文；需要时再读官方代码、作者前作、OpenReview。PDF 可缓存到 `runs/_paper/`。
2. **先预测，再揭晓**：在展示任何重建之前，按 LEARNING.md 的当前阶段向用户提 2–3 个问题（问题库在 skill reference 中）。把用户的原话和日期写进 DISCOVERY §1。用户说跳过就写"用户跳过"，不追问。
3. 写 DISCOVERY.md：
   - 推理链的每一环都标注 `[原文 §/Fig]`、`[作者前作/talk/OpenReview]` 或 `[推断]`，并给置信度。
   - 至少写一条备选发现路径和若干竞争解释。
   - 引用原文不超过 15 词；绝不编造作者的原话或经历。
4. 写 PLAN.md：
   - 必须自包含，因为执行者看不到聊天记录。
   - 冻结预测表要同时写用户和 agent 的预测，以及推翻条件。
   - 每个实验写清：旋钮、固定项、朴素预期、零信息基线、让效应消失的对照、指标、图规格、预算。
5. 用户批准后，把状态改为 planned，提交 `plan(<id>)`。此后不能再改预测；计划需要变更时，只在 PLAN.md 末尾追加"计划修订"一节（日期、原因、改动）。[H-000]

## 4. Executor

- 只依据本文件和 PLAN.md 工作，DISCOVERY.md 仅作背景。不修改冻结预测、DISCOVERY.md 和 REFLECTION.md。
- 预算：每个 topic 的计算总时长不超过 lab.toml 的 `topic_compute_minutes`（CPU/MPS）。L3（真实大模型、大数据、集群）只在 PLAN 明确写出时执行。单次下载超过 `download_gb_without_approval` 时先问用户。
- 先跑 smoke（1 个 seed、少量步数）确认管线正确，再正式运行；正式运行前把状态改为 running。
- 遇到以下情况就停下并写进 SUMMARY 的执行记录：耗时超过预算 1.5 倍；出现 NaN 或发散；发现与假设无关的 bug；PLAN 的歧义会影响结论。
- **不得为了得到预测的结果去调参、挑 seed 或改指标。**[H-000]
- 每个实验写 `results/E<n>.json`，至少包含 exp、status（done/partial/failed）、seeds、metrics、representative_seed、runtime_sec。
- 画图用 `tools/figstyle.py`，图存到 figures/；画完要实际打开看一遍。
- SUMMARY.md 的每一节都要写满。失败或未运行的实验照实写，不能写成已验证。
- `check --phase executed` 通过后，把状态改为 executed，提交 `run(<id>)`。

## 5. Reviewer（最好开新会话，不由执行者兼任）

1. **先核对**：SUMMARY 中的数字与 results/*.json 是否一致；逐张看图，确认读图说明成立。有错就改，并在执行记录里注明。
2. **反思对话**：和用户一起对照冻结预测，比较 DISCOVERY 的揭晓结果，讨论用到的招式和 taste 判断。taste 判断必须是用户自己的话；agent 只提问并给出自己的评分，不代写。
3. **合并更新**：
   - EXPERIMENTS.md：状态、一句话发现（≤40 字）、发现原型。
   - INSIGHTS.md、RESEARCH_MOVES.md。
   - LEARNING.md：校准表、当前阶段、下次练习重点。
4. 在 REFLECTION 的 Harness 复盘中记录摩擦指标；达到门槛时向 HARNESS_LOG.md 写提案（见 §7）。
5. `check --phase reviewed` 通过后，把状态改为 reviewed，提交 `review(<id>)`。

## 6. 实验与证据 [H-000]

- **toy 层级**：L0 纸笔/解析；L1 合成 toy（默认，单次运行 ≤5 分钟）；L2 小真实数据或小预训练模型；L3 真实复现（需要用户明确要求）。
- **实验顺序**：E1 先亲历"惊讶"，复现关键观察；E2 操纵假设中的因素；E3 及之后（可选）检验竞争解释。每个实验一次只改一个旋钮。
- **seed**：L1 至少 3 个 seed，报告 mean 和 min–max。定性图用代表性 seed（主指标最接近中位数的那个），另配一张全 seed 汇总图。偏离的 seed 只有在会改变结论时才展示，并标为反例。
- **结论分级**：严格区分"论文声称""toy 实测""推断"。toy 结果不等于复现了论文，SUMMARY 必须写明这个 toy 不能说明什么。
- **图**：图内文字用英文，中文解释写在 SUMMARY 的读图说明里。标题写结论，不写"Figure 1"。不用双 y 轴。

## 7. 规则如何改进（RSI）[H-000]

- agent 只提案，不直接修改 AGENTS.md、CLAUDE.md、lab.toml、templates/ 和 tools/。
- 提案写到 HARNESS_LOG.md 的"待批提案"下，必须引用具体 topic 中的事故证据，并优先修改或删除已有规则，而不是新增。
- 用户批准后才应用：规则末尾标 `[H-xxx]`，在 HARNESS_LOG 中记下复查期（2 个 topic 之后）；复查无效就撤销。
- 每复盘完 5 个 topic，Reviewer 提议做一次 retro：删除从未触发的规则、检查各文件容量、看预测校准趋势、挑出可以上游到 skill 的通用改进。

## 8. 环境

- Python 用 `/Users/max/codebase/.ml/.venv/bin/python`，运行代码和 tools 时显式写出这个路径。不要假设有 CUDA。安装依赖前先说明，并把版本记录到 SUMMARY 的复现一节。
- 资产路径见 lab.toml `[env]`。readonly_assets 中已下载的权重和数据只读复用，不修改、不删除、不重复下载。
