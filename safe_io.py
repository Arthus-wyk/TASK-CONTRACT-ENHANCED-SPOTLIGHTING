from __future__ import annotations

import sys
from typing import Any, TextIO


def safe_print(value: Any = "", *, file: TextIO | None = None, flush: bool = False) -> None:
    """
    Print without letting console encoding issues interrupt a benchmark run.

    Windows terminals commonly default to GBK. Some benchmark payloads contain
    characters that GBK cannot encode, so fall back to escaped text only when a
    normal print fails.
    """
    target = file or sys.stdout
    text = str(value)
    try:
        print(text, file=target, flush=flush)
    except UnicodeEncodeError:
        encoding = getattr(target, "encoding", None) or "ascii"
        escaped_text = text.encode(encoding, errors="backslashreplace").decode(
            encoding,
            errors="replace",
        )
        print(escaped_text, file=target, flush=flush)
