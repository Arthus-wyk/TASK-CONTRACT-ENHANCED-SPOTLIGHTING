from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_AUTO_MODEL = "ollama_cloud:gpt-oss-20b"
DEFAULT_AUTO_DEFENSE = "spotlighting"
SUPPORTED_DEFENSES = (
    "no_defense",
    "spotlighting",
    "task_shield",
    "task-shield",
    "spotlighting_task_shield",
    "spotlighting-task-shield",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start main.py as a detached background process."
    )
    parser.add_argument("--run-dir", default=".run")
    parser.add_argument(
        "--model",
        default=os.getenv("AGENTDOJO_AUTO_MODEL", DEFAULT_AUTO_MODEL),
        help=f"Model passed to main.py. Defaults to {DEFAULT_AUTO_MODEL}.",
    )
    parser.add_argument(
        "--defense",
        default=os.getenv("AGENTDOJO_AUTO_DEFENSE", DEFAULT_AUTO_DEFENSE),
        choices=SUPPORTED_DEFENSES,
        help=f"Defense mode passed to main.py. Defaults to {DEFAULT_AUTO_DEFENSE}.",
    )
    parser.add_argument(
        "--suite",
        action="append",
        default=[],
        help="Suite to run. Can be passed multiple times.",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=[],
        help="Suites to run. Overrides AGENTDOJO_AUTO_SUITES when provided.",
    )
    parser.add_argument("app_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = (repo_root / args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    pid_file = run_dir / "main.pid"
    stdout_file = run_dir / "main.out.log"
    stderr_file = run_dir / "main.err.log"

    if pid_file.exists():
        pid_text = pid_file.read_text(encoding="utf-8").strip()
        if pid_text and _is_running(int(pid_text)):
            print(f"Already running with PID {pid_text}.")
            print(f"Stdout: {stdout_file}")
            print(f"Stderr: {stderr_file}")
            return 0

    app_args = args.app_args
    if app_args and app_args[0] == "--":
        app_args = app_args[1:]

    configured_args = _configured_main_args(args, app_args)
    command = [sys.executable, "-u", "main.py", *configured_args, *app_args]
    creationflags = 0
    if sys.platform == "win32":
        creationflags = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )

    with stdout_file.open("ab") as stdout, stderr_file.open("ab") as stderr:
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
            creationflags=creationflags,
            start_new_session=sys.platform != "win32",
        )

    pid_file.write_text(str(process.pid), encoding="utf-8")
    print(f"Started background run with PID {process.pid}.")
    print(f"Command: {' '.join(command)}")
    print(f"Stdout: {stdout_file}")
    print(f"Stderr: {stderr_file}")
    print(f"PID file: {pid_file}")
    return 0


def _configured_main_args(args: argparse.Namespace, app_args: list[str]) -> list[str]:
    configured_args: list[str] = []

    if not _has_option(app_args, "--model"):
        configured_args.extend(["--model", args.model])
    if not _has_option(app_args, "--defense"):
        configured_args.extend(["--defense", args.defense])

    suites = [*args.suite, *args.suites]
    if not suites:
        suites = _split_suites(os.getenv("AGENTDOJO_AUTO_SUITES", ""))
    if suites and not _has_option(app_args, "--suites"):
        configured_args.extend(["--suites", *suites])

    return configured_args


def _split_suites(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def _has_option(args: list[str], option: str) -> bool:
    option_prefix = f"{option}="
    return any(arg == option or arg.startswith(option_prefix) for arg in args)


def _is_running(pid: int) -> bool:
    if sys.platform == "win32":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        return str(pid) in result.stdout

    try:
        import os

        os.kill(pid, 0)
    except OSError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
