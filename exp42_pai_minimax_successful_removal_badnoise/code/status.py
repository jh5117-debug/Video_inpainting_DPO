"""Status helpers for Exp42."""

from __future__ import annotations

EXP42_STATUS = "EXP42_PAI_MINIMAX_DATA_READBACK_COMPLETED"


def current_status() -> str:
    """Return the current Exp42 status string."""

    return EXP42_STATUS
