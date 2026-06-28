#!/usr/bin/env python3
"""Print the Exp38 scaffold status."""

from __future__ import annotations

from exp38_minimax_full_adapter_breakthrough.code.status import (
    BASE_BRANCH,
    BRANCH,
    EXPERIMENT_ID,
    READBACK_STATUS,
)


def main() -> None:
    print(f"experiment={EXPERIMENT_ID}")
    print(f"branch={BRANCH}")
    print(f"base_branch={BASE_BRANCH}")
    print(f"status={READBACK_STATUS}")


if __name__ == "__main__":
    main()

