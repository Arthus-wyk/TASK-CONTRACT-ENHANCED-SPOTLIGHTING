from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stop the background main.py run.")
    parser.add_argument("--run-dir", default=".run")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = (repo_root / args.run_dir).resolve()
    pid_file = run_dir / "main.pid"

    if not pid_file.exists():
        print(f"No PID file found at {pid_file}.")
        return 0

    pid_text = pid_file.read_text(encoding="utf-8").strip()
    if not pid_text:
        print("PID file is already empty.")
        return 0

    pid = int(pid_text)
    if not _is_running(pid):
        pid_file.write_text("", encoding="utf-8")
        print(f"Process {pid} is not running. Cleared stale PID file.")
        return 0

    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False)
    else:
        os.killpg(pid, signal.SIGTERM)

    pid_file.write_text("", encoding="utf-8")
    print(f"Stopped background run with PID {pid}.")
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
        os.kill(pid, 0)
    except OSError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
