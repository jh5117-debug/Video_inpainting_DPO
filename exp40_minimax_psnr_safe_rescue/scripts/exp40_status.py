#!/usr/bin/env python3
"""Print the current Exp40 status."""

from exp40_minimax_psnr_safe_rescue.code.status import current_status


if __name__ == "__main__":
    print(current_status())
