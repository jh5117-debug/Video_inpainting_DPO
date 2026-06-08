#!/usr/bin/env python
"""Process-title wrapper for Stage1 DPO training."""

from __future__ import annotations

import ctypes
import os
import runpy
import sys
from pathlib import Path


def set_process_title(name: str) -> None:
    try:
        import setproctitle

        setproctitle.setproctitle(name)
    except Exception:
        pass
    try:
        libc = ctypes.CDLL(None)
        libc.prctl(15, name[:15].encode("utf-8"), 0, 0, 0)
    except Exception:
        pass


def main() -> int:
    name = os.environ.setdefault("LINGBOT_PROCESS_NAME", "lingbot-worldmodel")
    os.environ.setdefault("PROCESS_TITLE", name)
    set_process_title(name)
    target = Path(__file__).with_name("train_stage1.py")
    sys.argv[0] = name
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
