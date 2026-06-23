#!/usr/bin/env python3
"""Canonical manifest schema for VOR object-removal preference rows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


TRAIN_SPLITS = {"train", "search_dev", "shadow_dev"}


@dataclass(frozen=True)
class VorORTripletRow:
    sample_id: str
    split: str
    task: str
    condition_video_path: str
    winner_video_path: str
    mask_path: str
    condition_source_role: str = "V_obj"
    winner_source_role: str = "V_bg"
    mask_source_role: str = "foreground_object_mask"
    hard_comp: bool = False
    comp_mode: str = "none"
    frame_indices: list[int] = field(default_factory=list)
    effect_type: str = ""
    source_type: str = ""

    @classmethod
    def from_dict(cls, row: dict) -> "VorORTripletRow":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        data = {k: row.get(k) for k in known if k in row}
        if isinstance(data.get("hard_comp"), str):
            data["hard_comp"] = data["hard_comp"].lower() in {"1", "true", "yes"}
        if data.get("frame_indices") in (None, ""):
            data["frame_indices"] = []
        return cls(**data)

    def validate(self, *, allow_eval: bool = False) -> list[str]:
        errors: list[str] = []
        if self.task != "object_removal":
            errors.append("task_must_be_object_removal")
        if self.condition_source_role != "V_obj":
            errors.append("condition_must_come_from_v_obj")
        if self.winner_source_role != "V_bg":
            errors.append("winner_must_be_v_bg")
        if self.mask_source_role != "foreground_object_mask":
            errors.append("mask_must_be_foreground_object_mask")
        if self.hard_comp:
            errors.append("hard_comp_must_be_false")
        if self.comp_mode != "none":
            errors.append("comp_mode_must_be_none")
        if not allow_eval and self.split.lower() in {"vor_eval", "eval", "test", "vor-eval"}:
            errors.append("vor_eval_must_not_enter_training_or_dev_manifests")
        if self.condition_video_path == self.winner_video_path:
            errors.append("condition_path_equals_winner_path_gt_leakage_risk")
        for field_name in ["sample_id", "condition_video_path", "winner_video_path", "mask_path"]:
            if not getattr(self, field_name):
                errors.append(f"missing_{field_name}")
        return errors

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "task": self.task,
            "condition_video_path": self.condition_video_path,
            "winner_video_path": self.winner_video_path,
            "mask_path": self.mask_path,
            "condition_source_role": self.condition_source_role,
            "winner_source_role": self.winner_source_role,
            "mask_source_role": self.mask_source_role,
            "hard_comp": self.hard_comp,
            "comp_mode": self.comp_mode,
            "frame_indices": self.frame_indices,
            "effect_type": self.effect_type,
            "source_type": self.source_type,
        }


@dataclass(frozen=True)
class VorORPreferenceRow(VorORTripletRow):
    loser_video_path: str = ""
    generator_source: str = ""
    generator_seed: int = 0
    generator_trained_on_vor: bool = False
    loser_is_raw: bool = True
    affected_region_path: str = ""

    @classmethod
    def from_dict(cls, row: dict) -> "VorORPreferenceRow":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        data = {k: row.get(k) for k in known if k in row}
        for key in ["hard_comp", "generator_trained_on_vor", "loser_is_raw"]:
            if isinstance(data.get(key), str):
                data[key] = data[key].lower() in {"1", "true", "yes"}
        if data.get("frame_indices") in (None, ""):
            data["frame_indices"] = []
        if isinstance(data.get("generator_seed"), str) and data["generator_seed"]:
            data["generator_seed"] = int(data["generator_seed"])
        return cls(**data)

    def validate(self, *, allow_eval: bool = False) -> list[str]:
        errors = super().validate(allow_eval=allow_eval)
        if not self.loser_video_path:
            errors.append("missing_loser_video_path")
        if self.loser_video_path in {self.condition_video_path, self.winner_video_path}:
            errors.append("loser_path_copies_condition_or_winner")
        if not self.loser_is_raw:
            errors.append("loser_must_be_raw_no_comp")
        if self.generator_source == "EffectErase" and not self.generator_trained_on_vor:
            errors.append("effecterase_generator_must_record_trained_on_vor")
        return errors

    def to_dict(self) -> dict:
        out = super().to_dict()
        out.update(
            {
                "loser_video_path": self.loser_video_path,
                "generator_source": self.generator_source,
                "generator_seed": self.generator_seed,
                "generator_trained_on_vor": self.generator_trained_on_vor,
                "loser_is_raw": self.loser_is_raw,
                "affected_region_path": self.affected_region_path,
            }
        )
        return out


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_triplet_manifest(path: Path, *, allow_eval: bool = False) -> list[dict]:
    failures = []
    for idx, row in enumerate(read_jsonl(path)):
        errors = VorORTripletRow.from_dict(row).validate(allow_eval=allow_eval)
        if errors:
            failures.append({"line": idx + 1, "sample_id": row.get("sample_id", ""), "errors": errors})
    return failures


def validate_preference_manifest(path: Path, *, allow_eval: bool = False) -> list[dict]:
    failures = []
    for idx, row in enumerate(read_jsonl(path)):
        errors = VorORPreferenceRow.from_dict(row).validate(allow_eval=allow_eval)
        if errors:
            failures.append({"line": idx + 1, "sample_id": row.get("sample_id", ""), "errors": errors})
    return failures


def manifest_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
