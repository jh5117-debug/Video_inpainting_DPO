#!/usr/bin/env python3
"""Collate helpers for VOR OR DPO rows."""

from __future__ import annotations


def vor_or_collate(batch: list[dict]) -> dict:
    if not batch:
        return {}
    out: dict[str, list] = {key: [] for key in batch[0]}
    for item in batch:
        for key, value in item.items():
            out.setdefault(key, []).append(value)
    return out
