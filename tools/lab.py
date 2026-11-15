#!/usr/bin/env python3
"""discovery-lab workspace tool: init, new, check, status.

Standard library only; requires Python 3.11+ (tomllib). The canonical copy lives in
the research-discovery-lab skill; `init` copies it into <workspace>/tools/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import difflib
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
from pathlib import Path

if sys.version_info < (3, 11):
    sys.exit("lab.py 需要 Python ≥3.11（tomllib）；请用 lab.toml 中的 python 运行。")

import tomllib  # noqa: E402

TOPIC_FILES = ("DISCOVERY.md", "PLAN.md", "SUMMARY.md", "REFLECTION.md")
TOPIC_DIRS = ("code", "figures", "results", "runs")
ROOT_FILES = ("AGENTS.md", "CLAUDE.md", "README.md", "EXPERIMENTS.md", "INSIGHTS.md",
              "RESEARCH_MOVES.md", "LEARNING.md", "HARNESS_LOG.md", "MACHINES.md", "lab.toml")
MACHINE_FILE = Path("~/.config/lab/machine")
RUN_STATUSES = ("running", "done", "failed", "cancelled")
COMMIT_SUBJECT = re.compile(r"^(?P<kind>[a-z]+)(?:\[(?P<machine>[\w-]+)\])?(?:\((?P<topic>[^)]*)\))?:\s*(?P<text>.*)$")
UPGRADE_DIFF_FILES = ("AGENTS.md", "CLAUDE.md")  # rule files; content files are never diffed
PHASES = ("discovery", "planned", "running", "executed", "reviewed")
OTHER_STATES = ("blocked", "abandoned")
COMPLETE_AT = {"DISCOVERY.md": "planned", "PLAN.md": "planned",
               "SUMMARY.md": "executed", "REFLECTION.md": "reviewed"}
RESULT_STATUSES = ("done", "partial", "failed")
DEFAULT_LINES = {"AGENTS.md": 120, "CLAUDE.md": 25, "README.md": 60, "INSIGHTS.md": 150,
                 "DISCOVERY.md": 150, "PLAN.md": 200, "SUMMARY.md": 200, "REFLECTION.md": 80}
LARGE_TRACKED_BYTES = 5 * 1024 * 1024
FINDING_MAX_CHARS = 40

PLACEHOLDER = re.compile(r"【待填[^】]*】")
COMMENT = re.compile(r"<!--.*?-->", re.S)
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)")
SLUG = re.compile(r"^[a-z0-9][a-z0-9-]*$")
TOPIC_ID = re.compile(r"^\d{4}-\d{2}-\d{2}-[a-z0-9][a-z0-9-]*$")
RENAMES = {"gitignore.txt": ".gitignore"}


# ---------- helpers ----------

def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render(text: str, tokens: dict[str, str]) -> str:
    for key, value in tokens.items():
        text = text.replace("{{%s}}" % key, value)
    return text


def strip_comments(text: str) -> str:
    return COMMENT.sub("", text)


def section(text: str, heading: str) -> str | None:
    match = re.search(rf"^## {re.escape(heading)}[ \t]*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    return match.group(1) if match else None


def normalized(text: str) -> str:
    return "\n".join(line.strip() for line in strip_comments(text).splitlines() if line.strip())


def git_toplevel(root: Path) -> Path | None:
    if not shutil.which("git"):
        return None
    result = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=root, capture_output=True, text=True)
    return Path(result.stdout.strip()).resolve() if result.returncode == 0 else None


def git(top: Path, *args: str) -> subprocess.CompletedProcess:
    # cwd instead of `git -C`, so old git (1.8 on the cluster) works too
    return subprocess.run(["git", *args], cwd=top, capture_output=True, text=True)


def current_machine() -> str:
    """Machine name from $LAB_MACHINE or ~/.config/lab/machine (mac / pc / cluster)."""
    name = os.environ.get("LAB_MACHINE", "").strip()
    path = MACHINE_FILE.expanduser()
    if not name and path.is_file():
        name = path.read_text(encoding="utf-8").strip()
    return name or "unknown"


def machine_config(cfg: dict, name: str) -> dict:
    return cfg.get("machines", {}).get(name, {})


def commit_and_push(root: Path, message: str, paths: list[Path], push: bool, allow_empty: bool = False) -> None:
    top = git_toplevel(root)
    if top is None:
        sys.exit("工作区不是 git 仓库，无法记录。")
    if paths:
        result = git(top, "add", "--", *[str(p) for p in paths])
        if result.returncode != 0:
            sys.exit(result.stderr.strip())
    args = ["commit", "-q", "-m", message] + (["--allow-empty"] if allow_empty else [])
    result = git(top, *args)
    if result.returncode != 0:
        sys.exit(f"提交失败：{(result.stderr or result.stdout).strip()}")
    print(f"已提交：{message}")
    remotes = git(top, "remote").stdout.split()
    if push and remotes:
        result = git(top, "push", "-q", remotes[0], "HEAD")
        print("已推送。" if result.returncode == 0 else f"WARN 推送失败，请稍后手动 git push：{result.stderr.strip()}")


def run_records(root: Path) -> list[tuple[Path, dict]]:
    records = []
    for path in sorted(root.glob("topics/*/results/runs/*.json")):
        try:
            records.append((path, json.loads(read(path))))
        except (json.JSONDecodeError, UnicodeDecodeError):
            records.append((path, {"run_id": path.stem, "status": "invalid"}))
    return records


def skill_dir(root: Path | None) -> Path:
    here = Path(__file__).resolve().parent.parent
    if (here / "assets" / "workspace").is_dir():
        return here
    if root is not None and (root / "lab.toml").is_file():
        path = load_config(root).get("skill", {}).get("path")
        if path and (Path(path) / "assets" / "workspace").is_dir():
            return Path(path)
    sys.exit("找不到 skill 的 assets；请在 lab.toml [skill].path 中填写 research-discovery-lab 的目录。")


def find_root(arg: str | None) -> Path:
    if arg:
        root = Path(arg).expanduser().resolve()
        if (root / "lab.toml").is_file():
            return root
        sys.exit(f"{root} 不是 discovery-lab 工作区（缺 lab.toml）。")
    cwd = Path.cwd().resolve()
    for path in (cwd, *cwd.parents, Path(__file__).resolve().parent.parent):
        if (path / "lab.toml").is_file():
            return path
    sys.exit("找不到工作区（向上查找 lab.toml 失败）；请用 --root 指定。")


def load_config(root: Path) -> dict:
    return tomllib.loads(read(root / "lab.toml"))


def limits(cfg: dict) -> dict:
    lim = cfg.get("limits", {})
    return {
        "lines": {**DEFAULT_LINES, **lim.get("lines", {})},
        "hard": set(lim.get("hard", ["AGENTS.md", "CLAUDE.md"])),
        "moves": int(lim.get("moves", 40)),
        "summary_figures": int(lim.get("summary_figures", 8)),
        "figure_kb": int(lim.get("figure_kb", 1024)),
    }


def split_row(line: str) -> list[str]:
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))]


def index_rows(root: Path) -> dict[str, list[str]]:
    """Topic rows of EXPERIMENTS.md keyed by topic id: [date, id, paper, archetype, status, finding, links]."""
    rows: dict[str, list[str]] = {}
    path = root / "EXPERIMENTS.md"
    if not path.is_file():
        return rows
    for line in read(path).splitlines():
        if line.startswith("|"):
            cells = split_row(line)
            if len(cells) >= 6 and TOPIC_ID.match(cells[1]):
                rows[cells[1]] = cells
    return rows


def topic_dirs(root: Path) -> list[Path]:
    base = root / "topics"
    return sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")) if base.is_dir() else []


def resolve_topic(root: Path, arg: str) -> Path:
    raw = Path(arg.rstrip("/"))
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, root / raw, root / "topics" / raw]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_dir() and candidate.parent == (root / "topics").resolve():
            return candidate
    sys.exit(f"找不到 topic：{arg}（应为 topics/<id> 或 <id>）")


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, scope: str, message: str) -> None:
        self.errors.append(f"ERROR {scope}: {message}")

    def warn(self, scope: str, message: str) -> None:
        self.warnings.append(f"WARN  {scope}: {message}")


# ---------- init ----------

def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve()
    skill = skill_dir(root)
    python = args.python or sys.executable
    tokens = {"SKILL_PATH": str(skill), "PYTHON": python,
              "DATE": dt.date.today().isoformat(), "WORKSPACE": root.name}
    planned: list[tuple[str, Path, bool]] = []  # (text, destination, diff on upgrade)
    for src in sorted((skill / "assets" / "workspace").iterdir()):
        if src.is_file():
            name = RENAMES.get(src.name, src.name)
            planned.append((render(read(src), tokens), root / name, name in UPGRADE_DIFF_FILES))
    for src in sorted((skill / "assets" / "topic").glob("*.md")):
        planned.append((read(src), root / "templates" / src.name, True))  # tokens filled by `new`
    for name in ("lab.py", "figstyle.py"):
        planned.append((read(skill / "scripts" / name), root / "tools" / name, True))

    for text, dest, diffable in planned:
        rel = dest.relative_to(root)
        if dest.exists():
            current = read(dest)
            if args.upgrade and diffable and current != text:
                sys.stdout.writelines(difflib.unified_diff(
                    current.splitlines(True), text.splitlines(True),
                    f"workspace/{rel}", f"skill/{rel}"))
            elif not args.upgrade:
                print(f"保留 {rel}")
            continue
        write(dest, text)
        print(f"创建 {rel}")
    (root / "topics").mkdir(exist_ok=True)
    if git_toplevel(root) is None and shutil.which("git"):
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        print("git init")
    if args.upgrade:
        print("upgrade 只打印差异，不覆盖文件；是否同步由用户决定。")
    return 0


# ---------- new ----------

def cmd_new(args: argparse.Namespace) -> int:
    root = find_root(args.root)
    cfg = load_config(root)
    if not SLUG.match(args.slug):
        sys.exit("slug 只能包含小写字母、数字和连字符，且以字母或数字开头。")
    date = dt.date.fromisoformat(args.date).isoformat() if args.date else dt.date.today().isoformat()
    templates = {name: root / "templates" / name for name in TOPIC_FILES}
    missing = [name for name, path in templates.items() if not path.is_file()]
    if missing:
        sys.exit(f"缺模板 {', '.join(missing)}；请运行 lab.py init 补齐。")
    index = root / "EXPERIMENTS.md"
    lines = read(index).splitlines()
    table = [i for i, line in enumerate(lines) if line.startswith("|")]
    if not table:
        sys.exit("EXPERIMENTS.md 中找不到索引表格。")

    base = f"{date}-{args.slug}"
    topic_id, n = base, 2
    while (root / "topics" / topic_id).exists():
        topic_id, n = f"{base}-{n}", n + 1
    topic = root / "topics" / topic_id
    for name in TOPIC_DIRS:
        (topic / name).mkdir(parents=True)
    tokens = {"TOPIC_ID": topic_id, "DATE": date, "PAPER": args.paper, "SLUG": args.slug,
              "PYTHON": cfg.get("env", {}).get("python", "python3")}
    for name, path in templates.items():
        write(topic / name, render(read(path), tokens))

    paper = args.paper.replace("|", "\\|")
    links = " · ".join(f"[{label}](topics/{topic_id}/{name})" for label, name in
                       (("发现", "DISCOVERY.md"), ("计划", "PLAN.md"), ("总结", "SUMMARY.md")))
    lines.insert(table[-1] + 1, f"| {date} | {topic_id} | {paper} | 【待填】 | discovery | 【待填】 | {links} |")
    write(index, "\n".join(lines) + "\n")
    print(f"创建 topics/{topic_id}/ 并登记到 EXPERIMENTS.md（状态 discovery）。")
    print("下一步：先按 LEARNING.md 的当前阶段向用户提预测问题，再写 DISCOVERY.md。")
    return 0


# ---------- check ----------

def line_budget(scope: str, path: Path, lim: dict, rep: Report) -> None:
    allowed = lim["lines"].get(path.name)
    if allowed is None or not path.is_file():
        return
    count = len(read(path).splitlines())
    if count > allowed:
        message = f"{path.name} 有 {count} 行，超过上限 {allowed}；请合并或删减"
        (rep.error if path.name in lim["hard"] and scope == "workspace" else rep.warn)(scope, message)


def check_workspace(root: Path, lim: dict, rep: Report) -> None:
    scope = "workspace"
    for name in ROOT_FILES:
        if not (root / name).is_file():
            rep.error(scope, f"缺少 {name}")
    for name in TOPIC_FILES:
        if not (root / "templates" / name).is_file():
            rep.error(scope, f"缺少 templates/{name}")
    if not (root / "topics").is_dir():
        rep.error(scope, "缺少 topics/ 目录")
    for name in ROOT_FILES:
        line_budget(scope, root / name, lim, rep)
    claude = root / "CLAUDE.md"
    if claude.is_file() and "@AGENTS.md" not in read(claude):
        rep.error(scope, "CLAUDE.md 必须用 @AGENTS.md 导入永久规则")
    moves = root / "RESEARCH_MOVES.md"
    if moves.is_file():
        count = len(re.findall(r"^### M\d+", read(moves), re.M))
        if count > lim["moves"]:
            rep.warn(scope, f"RESEARCH_MOVES.md 有 {count} 条招式，超过上限 {lim['moves']}；请合并相近的招式")

    rows = index_rows(root)
    dirs = {p.name for p in topic_dirs(root)}
    for name in sorted(dirs - rows.keys()):
        rep.error(scope, f"topics/{name} 没有登记在 EXPERIMENTS.md 中")
    for name in sorted(rows.keys() - dirs):
        rep.error(scope, f"EXPERIMENTS.md 登记了 {name}，但 topics/ 下没有这个目录")
    for name, cells in rows.items():
        if cells[4] not in PHASES + OTHER_STATES:
            rep.error(scope, f"{name} 的状态 {cells[4]!r} 无效，应为 {'/'.join(PHASES + OTHER_STATES)}")
        finding = cells[5]
        if "【待填" not in finding and len(finding) > FINDING_MAX_CHARS:
            rep.warn(scope, f"{name} 的一句话发现有 {len(finding)} 字，超过 {FINDING_MAX_CHARS} 字")


def check_frozen_predictions(root: Path, topic: Path, status: str, rep: Report) -> None:
    scope = f"topics/{topic.name}"
    plan = topic / "PLAN.md"
    if not plan.is_file():
        return
    current = section(read(plan), "冻结预测")
    if current is None:
        rep.error(scope, "PLAN.md 缺少“## 冻结预测”一节")
        return
    top = git_toplevel(root)
    if top is None:
        rep.warn(scope, "工作区不是 git 仓库，无法核对冻结预测")
        return
    rel = plan.resolve().relative_to(top).as_posix()
    log = git(top, "log", "--format=%H%x09%s", "--", rel).stdout.splitlines()
    marker = f"plan({topic.name})"
    pattern = re.compile(rf"^plan(\[[\w-]+\])?\({re.escape(topic.name)}\)")
    commits = [line.split("\t", 1)[0] for line in log if "\t" in line and pattern.match(line.split("\t", 1)[1])]
    if not commits:
        if status in PHASES and PHASES.index(status) >= PHASES.index("running"):
            rep.error(scope, f"预测未冻结：找不到 {marker} 提交")
        elif status == "planned":
            rep.warn(scope, f"用户批准后请提交 {marker}: … 以冻结预测")
        return
    frozen = git(top, "show", f"{commits[-1]}:{rel}")
    frozen_section = section(frozen.stdout, "冻结预测") if frozen.returncode == 0 else None
    if frozen_section is None or normalized(frozen_section) != normalized(current):
        rep.error(scope, f"冻结预测在 {marker} 提交（{commits[-1][:8]}）之后被修改；请恢复原文，变更写进“计划修订”")


def check_topic(root: Path, topic: Path, lim: dict, rep: Report, phase: str | None) -> None:
    scope = f"topics/{topic.name}"
    row = index_rows(root).get(topic.name)
    status = phase or (row[4] if row else "discovery")
    level = PHASES.index(status) if status in PHASES else -1  # blocked/abandoned: structure only

    for name in TOPIC_FILES:
        if not (topic / name).is_file():
            rep.error(scope, f"缺少 {name}")
    for name in TOPIC_DIRS:
        if not (topic / name).is_dir():
            rep.warn(scope, f"缺少 {name}/ 目录")
    for path in sorted(topic.rglob("*.md")):
        rel = path.relative_to(topic)
        if rel.parts[0] != "runs" and rel.as_posix() not in TOPIC_FILES:
            rep.error(scope, f"多余的 MD：{rel.as_posix()}（每个 topic 只允许 {'/'.join(TOPIC_FILES)}）")
    for name in TOPIC_FILES:
        line_budget(scope, topic / name, lim, rep)

    for name, needed in COMPLETE_AT.items():
        path = topic / name
        if level >= PHASES.index(needed) and path.is_file():
            found = PLACEHOLDER.findall(strip_comments(read(path)))
            if found:
                rep.error(scope, f"{name} 还有 {len(found)} 处【待填】（状态 {status} 要求完成），例如 {found[0]}")
    if row and level >= PHASES.index("planned") and "【待填" in row[3]:
        rep.error(scope, "EXPERIMENTS.md 中的发现原型未填写")
    if row and level >= PHASES.index("reviewed") and "【待填" in row[5]:
        rep.error(scope, "EXPERIMENTS.md 中的一句话发现未填写")

    for path in sorted((topic / "results" / "runs").glob("*.json")):
        try:
            record = json.loads(read(path))
        except (json.JSONDecodeError, UnicodeDecodeError):
            record = {}
        if not all(k in record for k in ("run_id", "machine", "status")) or record.get("status") not in RUN_STATUSES:
            rep.error(scope, f"运行记录 results/runs/{path.name} 需要 run_id、machine 和有效的 status")

    results = sorted((topic / "results").glob("*.json")) if (topic / "results").is_dir() else []
    for path in results:
        try:
            data = json.loads(read(path))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            rep.error(scope, f"results/{path.name} 不是合法 JSON：{error}")
            continue
        if not isinstance(data, dict) or "exp" not in data or data.get("status") not in RESULT_STATUSES:
            rep.error(scope, f"results/{path.name} 需要包含 exp，且 status 为 {'/'.join(RESULT_STATUSES)}")

    summary = topic / "SUMMARY.md"
    if level >= PHASES.index("executed") and summary.is_file():
        images = list(dict.fromkeys(m for m in IMAGE.findall(strip_comments(read(summary)))
                                    if not m.startswith(("http://", "https://"))))
        if not images:
            rep.error(scope, "SUMMARY.md 没有引用任何图")
        if len(images) > lim["summary_figures"]:
            rep.error(scope, f"SUMMARY.md 引用了 {len(images)} 张图，超过上限 {lim['summary_figures']}")
        for image in images:
            path = (topic / image).resolve()
            if not path.is_file():
                rep.error(scope, f"SUMMARY.md 引用的图不存在：{image}")
            elif path.stat().st_size > lim["figure_kb"] * 1024:
                rep.error(scope, f"{image} 有 {path.stat().st_size // 1024} KB，超过 {lim['figure_kb']} KB")
            elif path.parent != (topic / "figures").resolve():
                rep.warn(scope, f"{image} 不在 figures/ 中")
        if not results:
            rep.error(scope, "results/ 中没有 E*.json")

    for name in ("code", "figures", "results"):
        base = topic / name
        for path in base.rglob("*") if base.is_dir() else []:
            if path.is_file() and path.stat().st_size > LARGE_TRACKED_BYTES:
                rep.warn(scope, f"{path.relative_to(topic).as_posix()} 超过 5 MB，会进入 git；大文件放 runs/")

    top = git_toplevel(root)
    if top is not None:
        probe = (topic / "runs" / ".probe").resolve().relative_to(top).as_posix()
        if git(top, "check-ignore", "-q", probe).returncode != 0:
            rep.error(scope, "runs/ 没有被 git 忽略；检查 .gitignore 中的 topics/*/runs/")
    if level >= PHASES.index("planned") or level == -1:
        check_frozen_predictions(root, topic, status, rep)


def cmd_check(args: argparse.Namespace) -> int:
    root = find_root(args.root)
    lim = limits(load_config(root))
    rep = Report()
    check_workspace(root, lim, rep)
    if args.topic:
        topics = [resolve_topic(root, args.topic)]
    elif args.phase:
        sys.exit("--phase 需要同时指定 topic。")
    else:
        topics = topic_dirs(root)
    for topic in topics:
        check_topic(root, topic, lim, rep, args.phase)
    for line in rep.errors + rep.warnings:
        print(line)
    print(f"check：{len(rep.errors)} 个错误，{len(rep.warnings)} 个警告")
    return 1 if rep.errors else 0


# ---------- runs, log, journal ----------

def cmd_run_start(args: argparse.Namespace) -> int:
    root = find_root(args.root)
    cfg = load_config(root)
    topic = resolve_topic(root, args.topic)
    command = " ".join(args.cmd).strip()
    if not command:
        sys.exit("请在 -- 之后给出要运行的命令。")
    me = current_machine()
    if me == "unknown":
        sys.exit("不知道本机是哪台：请写 ~/.config/lab/machine（mac / pc / cluster）或设置 LAB_MACHINE。")
    top = git_toplevel(root)
    if top is None:
        sys.exit("工作区不是 git 仓库，无法记录运行。")
    code_rel = (topic / "code").resolve().relative_to(top).as_posix()
    if git(top, "status", "--porcelain", "--", code_rel).stdout.strip():
        sys.exit(f"{code_rel} 有未提交的改动：先提交代码，运行记录才能指向确定的版本。")
    commit = git(top, "rev-parse", "HEAD").stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{args.exp}-{stamp}-{me}"
    runs_root = machine_config(cfg, me).get("runs_root") or ""
    run_dir = (Path(runs_root).expanduser() / topic.name / run_id) if runs_root else topic / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    archive = subprocess.run(["git", "archive", "--format=tar", "HEAD", "--", code_rel],
                             cwd=top, capture_output=True)
    if archive.returncode != 0:
        sys.exit(archive.stderr.decode(errors="replace").strip())
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(run_dir, filter="data")
    record = {"run_id": run_id, "topic": topic.name, "exp": args.exp, "machine": me,
              "host": socket.gethostname(), "run_dir": str(run_dir.resolve()),
              "code_dir": str((run_dir / code_rel).resolve()), "command": command, "commit": commit,
              "started": dt.datetime.now().isoformat(timespec="seconds"), "ended": None,
              "status": "running", "job": args.job, "note": args.note}
    path = topic / "results" / "runs" / f"{run_id}.json"
    write(path, json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(f"运行 {run_id}：机器 {me}，目录 {run_dir}")
    print(f"代码快照（提交 {commit[:8]}）：{run_dir / code_rel}")
    commit_and_push(root, f"run[{me}]({topic.name}): start {run_id}", [path], push=not args.no_push)
    if not args.job:
        print(f"建议在 tmux 中启动：tmux new -d -s {run_id} 'cd {run_dir} && {command} 2>&1 | tee run.log'")
    return 0


def cmd_run_end(args: argparse.Namespace) -> int:
    root = find_root(args.root)
    matches = [(p, r) for p, r in run_records(root) if r.get("run_id") == args.run_id]
    if not matches:
        sys.exit(f"找不到运行记录 {args.run_id}；先 git pull 再试。")
    path, record = matches[0]
    record.update(status=args.status, ended=dt.datetime.now().isoformat(timespec="seconds"))
    if args.note:
        record["note"] = f"{record.get('note') or ''} {args.note}".strip()
    if args.outputs:
        record["outputs"] = args.outputs
    write(path, json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    me = current_machine()
    commit_and_push(root, f"run[{me}]({record.get('topic', '?')}): end {args.run_id} {args.status}",
                    [path], push=not args.no_push)
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    root = find_root(args.root)
    rows = [r for _, r in run_records(root)
            if (not args.active or r.get("status") == "running")
            and (not args.machine or r.get("machine") == args.machine)]
    if not rows:
        print("没有符合条件的运行记录（不同机器的记录要先 git pull 才能看到）。")
    for r in sorted(rows, key=lambda r: r.get("started") or ""):
        print(f"{r.get('status', '?'):<9} {r.get('machine', '?'):<8} {r.get('run_id')}  "
              f"{r.get('topic', '?')}  {r.get('job') or ''}  {r.get('run_dir', '')}")
    return 0


def cmd_log(args: argparse.Namespace) -> int:
    root = find_root(args.root)
    text = " ".join(args.text).strip()
    if not text:
        sys.exit("请写下做了什么、产物在哪。")
    commit_and_push(root, f"log[{current_machine()}]: {text}", [], push=not args.no_push, allow_empty=True)
    return 0


def cmd_journal(args: argparse.Namespace) -> int:
    root = find_root(args.root)
    top = git_toplevel(root)
    if top is None:
        sys.exit("工作区不是 git 仓库。")
    out = git(top, "log", f"--since={args.days}.days", "--date=iso", "--format=%ad%x09%s").stdout.splitlines()
    shown = 0
    for line in out:
        date, _, subject = line.partition("\t")
        date = date[:16]
        match = COMMIT_SUBJECT.match(subject)
        machine = (match.group("machine") if match else None) or "?"
        if args.machine and machine != args.machine:
            continue
        print(f"{date}  {machine:<8} {subject}")
        shown += 1
    if not shown:
        print(f"最近 {args.days} 天没有记录（其他机器的记录要先 git pull）。")
    return 0


# ---------- status ----------

def cmd_status(args: argparse.Namespace) -> int:
    root = find_root(args.root)
    rows = index_rows(root)
    learning = read(root / "LEARNING.md") if (root / "LEARNING.md").is_file() else ""
    stage = re.search(r"当前阶段[:：]\s*(S\d)", learning)
    focus = re.search(r"下次练习重点[:：]\s*(.+)", learning)
    log = read(root / "HARNESS_LOG.md") if (root / "HARNESS_LOG.md").is_file() else ""
    pending = len(re.findall(r"^### H-\d+", strip_comments(section(log, "待批提案") or ""), re.M))
    reviewed = sum(1 for cells in rows.values() if cells[4] == "reviewed")
    print(f"工作区：{root} · 本机：{current_machine()}")
    print(f"学习阶段：{stage.group(1) if stage else '?'} · 练习重点：{focus.group(1).strip() if focus else '—'}")
    print(f"待批 harness 提案：{pending} · 已复盘 topic：{reviewed}（每 5 个做一次 retro）")
    if not rows:
        print('还没有 topic。开始：lab.py new <slug> --paper "<标题或 arXiv>"')
    for topic_id, cells in sorted(rows.items()):
        print(f"  {cells[4]:<10} {topic_id}  {cells[2]}")
    active: dict[str, list[str]] = {}
    for _, r in run_records(root):
        if r.get("status") == "running":
            active.setdefault(r.get("machine", "?"), []).append(f"{r.get('run_id')}（{r.get('job') or r.get('run_dir', '')}）")
    print("正在进行的运行：" + ("无" if not active else ""))
    for machine, items in sorted(active.items()):
        print(f"  {machine}: " + "；".join(items))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lab.py", description="discovery-lab 工作区工具")
    sub = parser.add_subparsers(dest="command", required=True)
    p_init = sub.add_parser("init", help="在 root 创建缺失的工作区文件（绝不覆盖）")
    p_init.add_argument("root")
    p_init.add_argument("--python", help="写入规则和 lab.toml 的 Python 路径（默认当前解释器）")
    p_init.add_argument("--upgrade", action="store_true", help="另外打印规则、模板、工具与 skill 版本的差异（不写入）")
    p_new = sub.add_parser("new", help="新建 topic 并登记到 EXPERIMENTS.md")
    p_new.add_argument("slug")
    p_new.add_argument("--paper", required=True, help="论文标题或 arXiv 链接")
    p_new.add_argument("--date", help="YYYY-MM-DD，默认今天")
    p_new.add_argument("--root")
    p_check = sub.add_parser("check", help="检查工作区和 topic；有错误时以非零状态退出")
    p_check.add_argument("topic", nargs="?")
    p_check.add_argument("--phase", choices=PHASES + OTHER_STATES, help="按此状态检查，而不是 EXPERIMENTS.md 中的状态")
    p_check.add_argument("--root")
    p_status = sub.add_parser("status", help="打印 topic 状态、学习阶段、待批提案和正在进行的运行")
    p_status.add_argument("--root")
    p_run = sub.add_parser("run", help="记录一次运行：在执行机器上 start，结束后 end")
    run_sub = p_run.add_subparsers(dest="run_command", required=True)
    p_start = run_sub.add_parser("start", help="快照已提交的 topic 代码到 runs_root，写运行记录并提交",
                                 epilog="用法：lab.py run start topics/<id> --exp E1 -- <命令>（命令只记录，不执行）")
    p_start.add_argument("topic")
    p_start.add_argument("--exp", required=True, help="实验编号，例如 E1")
    p_start.add_argument("--job", help="Slurm 作业号或 tmux 会话名")
    p_start.add_argument("--note")
    p_start.add_argument("--no-push", action="store_true")
    p_start.add_argument("--root")
    p_end = run_sub.add_parser("end", help="标记运行结束并提交")
    p_end.add_argument("run_id")
    p_end.add_argument("--status", required=True, choices=RUN_STATUSES[1:])
    p_end.add_argument("--outputs", help="大产物所在位置（留在执行机器上）")
    p_end.add_argument("--note")
    p_end.add_argument("--no-push", action="store_true")
    p_end.add_argument("--root")
    p_runs = sub.add_parser("runs", help="列出所有机器上的运行记录")
    p_runs.add_argument("--active", action="store_true", help="只看正在进行的")
    p_runs.add_argument("--machine")
    p_runs.add_argument("--root")
    p_log = sub.add_parser("log", help="为零散工作留痕：生成一个带机器前缀的空提交")
    p_log.add_argument("text", nargs="+")
    p_log.add_argument("--no-push", action="store_true")
    p_log.add_argument("--root")
    p_journal = sub.add_parser("journal", help="按时间列出各机器上的提交")
    p_journal.add_argument("--days", type=int, default=7)
    p_journal.add_argument("--machine")
    p_journal.add_argument("--root")
    argv = list(sys.argv[1:] if argv is None else argv)
    tail: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, tail = argv[:cut], argv[cut + 1:]
    args = parser.parse_args(argv)
    args.cmd = tail
    if args.command == "run":
        return {"start": cmd_run_start, "end": cmd_run_end}[args.run_command](args)
    return {"init": cmd_init, "new": cmd_new, "check": cmd_check, "status": cmd_status,
            "runs": cmd_runs, "log": cmd_log, "journal": cmd_journal}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
