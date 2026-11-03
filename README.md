# 深度学习论文复现与 idea 验证

本工作目录按“项目 → 多轮实验”组织。工作规范集中维护在 [AGENTS.md](AGENTS.md)，实验总结使用 [统一模板](templates/EXPERIMENT_SUMMARY.md)。

## 启动新会话

```bash
cd /Users/max/codebase/coding_agent_repro
agy
```

Antigravity CLI 支持启动时自动读取工作根目录的 AGENTS.md，新会话无需重复粘贴环境与总结要求。新任务使用普通 `agy` 启动；`agy --continue` 或 `agy --conversation <ID>` 用于恢复已有会话。

规则文件是 agent 的持续工作指导，不能替代对实际执行与交付的检查。升级 CLI 或更换启动方式后，可在新会话中要求它只读说明当前工作规范，核对路径与总结要求。

参考：[Antigravity CLI best practices](https://antigravity.google/docs/cli/best-practices/)。

## 共享环境与资源

| 用途 | 固定位置 |
| --- | --- |
| 虚拟环境 | `/Users/max/codebase/.ml/.venv` |
| Python | `/Users/max/codebase/.ml/.venv/bin/python` |
| 下载模型 | `/Users/max/codebase/brainstorm/assets/models` |
| 下载数据集 | `/Users/max/codebase/brainstorm/assets/datasets` |

运行命令明确指定 Python，例如：

```bash
/Users/max/codebase/.ml/.venv/bin/python <脚本路径> <参数>
```

先复用共享资源，再按需下载。配置中显式记录实际模型与数据路径；原始共享资源保持完整。训练生成的 checkpoint、日志、指标与图表保存在实验目录。缺少依赖时检查兼容性并记录变更，不擅自重建共享环境。

## 目录约定

```text
coding_agent_repro/
├── AGENTS.md
├── README.md
├── templates/
│   └── EXPERIMENT_SUMMARY.md
└── projects/
    └── <论文或idea名称>/
        ├── README.md
        ├── src/
        ├── configs/
        └── experiments/
            └── <YYYYMMDD-HHMMSS-实验名称>/
                ├── SUMMARY.md
                ├── config.yaml
                ├── logs/
                └── outputs/
```

项目和实验子目录在实际任务开始时创建；时间戳使用本地时间。同一项目共用代码，每轮实验独立保存最终配置和产物。继续同一实验时更新其总结，新的对比实验使用新目录。

项目 README.md 维护目标、来源、代码入口和实验索引。索引建议包含：实验名称、状态、验证目标、关键结果、SUMMARY.md 相对链接。

## 一轮实验的交付

1. 明确研究问题、假设、baseline 和成功标准，查看相关历史总结。
2. 建立实验目录，从总结模板填写 motivation 和计划设置。
3. 保存实际配置，运行并保留命令、代码版本、环境信息、日志和结果。
4. 补全 SUMMARY.md：为什么做、怎么做、如何复现、实际结果、能得出什么结论。
5. 更新项目实验索引，再向用户报告结果与总结位置。

失败、提前停止或只有 smoke test 结果也需要总结，状态必须如实标注。未执行的内容不能写成已验证结论。
