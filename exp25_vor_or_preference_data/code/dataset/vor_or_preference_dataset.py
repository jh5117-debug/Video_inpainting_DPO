#!/usr/bin/env python3
"""Lightweight manifest-backed VOR OR preference dataset.

Actual video decoding is delegated to the DiffuEraser trainer/view cache.  This
adapter owns schema semantics so OR cannot accidentally reuse BR's
winner-derived conditioning.
"""

from __future__ import annotations

from pathlib import Path

from .vor_or_manifest_schema import VorORPreferenceRow, read_jsonl


class VorORPreferenceDataset:
    def __init__(self, manifest_path: str | Path, *, validate: bool = True):
        self.manifest_path = Path(manifest_path)
        self.rows = [VorORPreferenceRow.from_dict(row) for row in read_jsonl(self.manifest_path)]
        if validate:
            failures = []
            for idx, row in enumerate(self.rows):
                errors = row.validate()
                if errors:
                    failures.append((idx, row.sample_id, errors))
            if failures:
                raise ValueError(f"Invalid VOR OR preference manifest rows: {failures[:5]}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        return {
            "condition_video_path": row.condition_video_path,
            "winner_video_path": row.winner_video_path,
            "loser_video_path": row.loser_video_path,
            "mask_path": row.mask_path,
            "affected_region_path": row.affected_region_path,
            "sample_id": row.sample_id,
            "frame_indices": row.frame_indices,
            "generator_source": row.generator_source,
            "effect_type": row.effect_type,
            "source_type": row.source_type,
            "task": row.task,
            "hard_comp": row.hard_comp,
            "comp_mode": row.comp_mode,
        }
