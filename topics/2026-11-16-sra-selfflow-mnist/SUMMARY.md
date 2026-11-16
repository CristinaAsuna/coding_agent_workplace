# From SRA to Self-Flow — MNIST L2：执行记录

状态：blocked；后台阶段 E1 停止。

累计 1189.07 秒（含 smoke）；原因：`Command '['/home/pc/lab/envs/main/bin/python', '/home/pc/lab/runs/2026-11-16-sra-selfflow-mnist/E1-20261116-071540-pc/topics/2026-11-16-sra-selfflow-mnist/code/run.py', '--exp', 'E1', '--seeds', '0', '--data-root', '/mnt/d/Datasets', '--out', '/home/pc/lab/runs/2026-11-16-sra-selfflow-mnist/E1-20261116-071540-pc/outputs', '--classifier', '/home/pc/lab/coding_agent_repro/topics/2026-11-16-sra-selfflow-mnist/runs/smoke/classifier.pt', '--already-sec', '174.3929354090069', '--budget-sec', '36000']' returned non-zero exit status 1.`。运行数据位于 `/home/pc/lab/runs/2026-11-16-sra-selfflow-mnist/E1-20261116-071540-pc/outputs`。未标记 executed，结果和图尚待人工核对。

原 4 小时预算在 smoke 外推为 8.31 小时后触发停止，用户随后批准提高至 10 小时。数据只读复用 `/mnt/d/Datasets/MNIST/raw/`。
