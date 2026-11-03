# coding_agent_repro：科研发现复现实验室

读一篇论文时，复现作者"看到了什么 → 怎么想 → 做了什么实验"，用 toy 实验和图亲手走一遍，借此练习科研判断。规则写在 [AGENTS.md](AGENTS.md)，所有 coding agent 都会读它；Claude Code 通过 [CLAUDE.md](CLAUDE.md) 导入这些规则。

## 一个 topic 的 5 步

1. **开题（Claude）**：在本目录对 Claude 说"用 research-discovery-lab 开一个 topic：<论文标题或 arXiv 链接>"。建议在 Plan mode 中进行。
2. **先预测**：回答 agent 提的 2–3 个问题，例如你觉得作者看到了什么、toy 的结果会怎样。回答之后才会看到完整的发现过程重建（DISCOVERY.md）。
3. **批准计划**：读 PLAN.md，确认后 agent 会提交 `plan(<id>)`，你的预测从此冻结，不再修改。
4. **执行（Codex）**：在 Codex 中说"按 AGENTS.md 执行 topics/<id>/PLAN.md"。
5. **复盘（Claude，开新会话）**：说"review topics/<id>"。对照预测、写下你自己的 taste 判断、更新根目录的总结文件；agent 可能会提出规则改进提案，由你批准。

每复盘完 5 个 topic，说一次"做 retro"。

## 文件

| 文件 | 作用 |
| --- | --- |
| [AGENTS.md](AGENTS.md) | 永久规则 |
| [EXPERIMENTS.md](EXPERIMENTS.md) | topic 索引与状态 |
| [INSIGHTS.md](INSIGHTS.md) | 跨 topic 的稳定认识与开放问题 |
| [RESEARCH_MOVES.md](RESEARCH_MOVES.md) | 科研招式库 |
| [LEARNING.md](LEARNING.md) | 你的成长档案：阶段、预测校准、练习重点 |
| [HARNESS_LOG.md](HARNESS_LOG.md) | 规则改进提案与记录（RSI） |
| lab.toml | 环境、预算、各文件容量上限 |
| templates/ · tools/ | topic 模板 · `lab.py`（new / check / status）与 `figstyle.py` |

每个 topic 在 `topics/<日期>-<slug>/` 下，固定包含 DISCOVERY / PLAN / SUMMARY / REFLECTION 四个文件。

## 常用命令

```bash
/Users/max/codebase/.ml/.venv/bin/python tools/lab.py status
```

```bash
/Users/max/codebase/.ml/.venv/bin/python tools/lab.py check
```

## 第一个 topic 可以选

优先选适合做 toy、发现故事清楚的论文：

- **Grokking**（Power et al., 2022）：过拟合很久之后，测试准确率突然上升。
- **Deep Double Descent**（Nakkiran et al., 2019）：测试误差随模型变大先升后降。
- **Edge of Stability**（Cohen et al., 2021）：全批量梯度下降时，sharpness 停在 2/η 附近。
- **Neural Collapse**（Papyan et al., 2020）：训练后期，类内特征塌缩到类均值，类均值构成单纯形。
