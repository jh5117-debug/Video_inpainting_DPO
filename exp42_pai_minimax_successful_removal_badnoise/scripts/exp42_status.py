#!/usr/bin/env python3
"""Print the current Exp42 status."""

from exp42_pai_minimax_successful_removal_badnoise.code.status import current_status


def main() -> None:
    print(current_status())


if __name__ == "__main__":
    main()
