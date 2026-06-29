"""Status helpers for Exp44."""

from __future__ import annotations

EXP44_STATUS = "EXP44_TARGETED_READBACK_COMPLETED"


def current_status() -> str:
    """Return the current Exp44 status string."""

    return EXP44_STATUS
