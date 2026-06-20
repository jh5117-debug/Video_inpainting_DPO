"""Best-effort process-title helpers for Exp23 PAI runs."""

from __future__ import annotations

import ctypes
import os


PR_SET_NAME = 15


def set_process_title(title: str = "Phy") -> dict[str, str]:
    status: dict[str, str] = {"requested": title}
    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(title)
        status["setproctitle"] = "ok"
    except Exception as exc:  # pragma: no cover - environment dependent
        status["setproctitle"] = f"failed:{type(exc).__name__}:{exc}"

    try:
        libc = ctypes.CDLL(None)
        rc = libc.prctl(PR_SET_NAME, ctypes.c_char_p(title.encode("utf-8")[:15]), 0, 0, 0)
        status["prctl"] = "ok" if rc == 0 else f"rc={rc}"
    except Exception as exc:  # pragma: no cover - environment dependent
        status["prctl"] = f"failed:{type(exc).__name__}:{exc}"

    os.environ["PROCESS_TITLE"] = title
    os.environ["SETPROCTITLE"] = title
    return status

