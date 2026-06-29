"""Status helpers for Exp42."""

from __future__ import annotations

EXP42_STATUS = "MINIMAX_SUCCESSFUL_REMOVAL_POOL_WEAK"


def current_status() -> str:
    """Return the current Exp42 status string."""

    return EXP42_STATUS
