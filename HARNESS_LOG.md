# Harness 改进记录（RSI）

agent 只负责写提案；用户批准后，才能修改 AGENTS.md、CLAUDE.md、lab.toml、templates/ 和 tools/。AGENTS.md 中的规则用 `[H-xxx]` 标注来源。方法细节见 skill reference `harness-rsi.md`。

## 待批提案

（暂无）

<!-- 每个提案一个小节，格式如下：
### H-001 一句话标题
- 来源：topics/<id>；证据：具体发生了什么
- 改动：文件 + 改动摘要（优先修改或删除已有规则）
- 预期效果 / 代价：……
- 复查方式：之后 2 个 topic 中看哪个信号
-->

## 已采纳

| 编号 | 日期 | 改动 | 复查时间 | 复查结论 |
|---|---|---|---|---|
| H-000 | 2026-11-03 | 初始规则，由 skill research-discovery-lab 生成 | 第 2 个 topic 复盘后 | — |
| H-001 | 2026-11-14 | 三机协作：MACHINES.md、lab.toml `[machines.*]`、`lab.py run/log/journal`、执行前告知（用户批准的计划） | 第 2 个跨机器运行后 | — |

## 已撤销 / 拒绝

（暂无）

## 摩擦指标

每个 topic 复盘时记录一行，retro 时据此看趋势。

| topic | 计划缺陷导致的返工 | 执行偏差 | 实际/预算耗时 | check 失败项 |
|---|---|---|---|---|

## Retro 记录

每复盘完 5 个 topic 做一次。

（暂无）
