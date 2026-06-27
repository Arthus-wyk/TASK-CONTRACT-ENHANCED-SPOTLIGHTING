from __future__ import annotations
from pathlib import Path
from dotenv import load_dotenv
import cyclopts

from runner import main

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)
if __name__ == "__main__":
    cyclopts.run(main)
