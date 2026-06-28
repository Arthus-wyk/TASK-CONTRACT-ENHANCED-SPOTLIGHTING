from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start main.py as a detached background process."
    )
    parser.add_argument("--run-dir", default=".run")
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

    command = [sys.executable, "-u", "main.py", *app_args]
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
        )

    pid_file.write_text(str(process.pid), encoding="utf-8")
    print(f"Started background run with PID {process.pid}.")
    print(f"Stdout: {stdout_file}")
    print(f"Stderr: {stderr_file}")
    print(f"PID file: {pid_file}")
    return 0


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
