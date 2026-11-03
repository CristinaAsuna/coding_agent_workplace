@AGENTS.md

# Claude 的补充

- Claude 默认担任 Planner 和 Reviewer，使用 skill `research-discovery-lab`，按角色读取它的 references。
- 在 Plan mode 中只能写 `~/.claude/plans/` 下的 plan 文件，它不是 topic 的 PLAN.md。因此在 Plan mode 里先读论文、提预测问题、起草 DISCOVERY 和 PLAN；ExitPlanMode 获批后，再运行 `tools/lab.py new`，把内容写进 topic 并提交 `plan(<id>)`。
- 提预测问题优先用 AskUserQuestion，允许用户自由回答，并记录原话。提问必须在展示重建之前完成。
- 用户要求执行实验时，默认交给 Codex 等执行者，并给出一句可直接粘贴的指令："按 AGENTS.md 执行 topics/<id>/PLAN.md"。只有用户明确要求时，才由 Claude 亲自按 AGENTS.md §4 执行，此时复盘应在另一个新会话中进行。
