#!/usr/bin/env python
"""Exp23 Stage1 entrypoint.

The full isolated implementation lives in ``train_stage1.py`` so Stage2 can
import the same DPO helpers without reaching into old experiment folders.
"""

from __future__ import annotations

from train_stage1 import main, parse_args


if __name__ == "__main__":
    main(parse_args())
