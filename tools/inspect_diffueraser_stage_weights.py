#!/usr/bin/env python
"""Inspect DiffuEraser Stage1/Stage2 checkpoint structure.

The goal is to make the Stage1-spatial / Stage2-temporal boundary explicit
before building DPO-S1 + SFT-S2 hybrid inference checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPATIAL_PREFIXES = (
    "conv_in",
    "time_proj",
    "time_embedding",
    "down_blocks",
    "up_blocks",
    "mid_block",
    "conv_norm_out",
    "conv_act",
    "conv_out",
)
TEMPORAL_TOKENS = (
    "motion",
    "temporal",
    "time_mixer",
    "motion_modules",
    "temporal_transformer",
    "temporal_attention",
)
WEIGHT_EXTS = {".safetensors", ".bin", ".pt", ".pth"}


@dataclass
class WeightDirReport:
    label: str
    path: str
    exists: bool
    files: List[str]
    configs: Dict[str, Dict[str, Any]]
    weight_files: List[str]
    unet_key_count: Optional[int]
    unet_key_sample: List[str]
    spatial_key_sample: List[str]
    temporal_key_sample: List[str]
    uncertain_key_sample: List[str]
    note: str = ""


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"_read_error": str(exc)}


def list_relative_files(root: Path, limit: int = 80) -> List[str]:
    if not root.exists():
        return []
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        files.append(path.relative_to(root).as_posix())
        if len(files) >= limit:
            files.append("...")
            break
    return files


def find_weight_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in WEIGHT_EXTS)


def sample_safetensors_keys(path: Path) -> Optional[List[str]]:
    try:
        from safetensors import safe_open
    except Exception:
        return None
    try:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            return list(f.keys())
    except Exception:
        return None


def sample_weight_keys(root: Path, limit: int = 160) -> tuple[Optional[int], List[str]]:
    unet_root = root / "unet_main"
    weights = find_weight_files(unet_root)
    if not weights:
        return None, []
    keys: Optional[List[str]] = None
    for weight in weights:
        if weight.suffix.lower() == ".safetensors":
            keys = sample_safetensors_keys(weight)
            if keys is not None:
                break
    if keys is None:
        try:
            import torch

            state = torch.load(str(weights[0]), map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
            keys = list(state.keys()) if isinstance(state, dict) else []
        except Exception:
            keys = []
    return len(keys), keys[:limit]


def classify_key(key: str) -> str:
    lowered = key.lower()
    if any(token in lowered for token in TEMPORAL_TOKENS):
        return "temporal"
    if key.startswith(SPATIAL_PREFIXES):
        return "spatial"
    return "uncertain"


def inspect_dir(label: str, path: Path) -> WeightDirReport:
    configs = {}
    for rel in ["unet_main/config.json", "brushnet/config.json", "scheduler/scheduler_config.json"]:
        cfg = read_json(path / rel)
        if cfg:
            configs[rel] = cfg

    key_count, key_sample = sample_weight_keys(path)
    spatial = [k for k in key_sample if classify_key(k) == "spatial"][:40]
    temporal = [k for k in key_sample if classify_key(k) == "temporal"][:40]
    uncertain = [k for k in key_sample if classify_key(k) == "uncertain"][:40]
    note = ""
    if path.exists() and not (path / "unet_main").exists():
        note = "path exists but is not exported as a DiffuEraser weights dir with unet_main/"

    return WeightDirReport(
        label=label,
        path=str(path),
        exists=path.exists(),
        files=list_relative_files(path),
        configs=configs,
        weight_files=[p.relative_to(path).as_posix() for p in find_weight_files(path)] if path.exists() else [],
        unet_key_count=key_count,
        unet_key_sample=key_sample,
        spatial_key_sample=spatial,
        temporal_key_sample=temporal,
        uncertain_key_sample=uncertain,
        note=note,
    )


def dedupe(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            out.append(path)
            seen.add(key)
    return out


def glob_many(patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(Path("/").glob(pattern.lstrip("/")))
    return dedupe(paths)


def default_stage1_candidates() -> List[Path]:
    return glob_many(
        [
            "/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/*exp7*stage1*/checkpoint-*",
            "/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/*exp7*stage1*/last_weights",
            "/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/stage1/*exp7*stage1*/checkpoint-*",
            "/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/stage1/*exp7*stage1*/last_weights",
        ]
    )


def default_stage2_candidates() -> List[Path]:
    return glob_many(
        [
            "/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/*exp7*stage2*/last_weights",
            "/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/stage2/*exp7*stage2*/last_weights",
        ]
    )


def default_sft_stage2_candidates() -> List[Path]:
    candidates: List[Path] = []
    for env_key in ["YOUTUBE_VOS_SFT_STAGE2_WEIGHTS", "SFT_STAGE2_WEIGHTS", "BASE_STAGE2_WEIGHTS"]:
        value = os.environ.get(env_key)
        if value:
            candidates.append(Path(value))
    candidates.extend(
        [
            Path("/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000"),
            Path("/mnt/nas/hj/weights/diffuEraser/converted_weights_step34000"),
            Path("/mnt/nas/hj/H20_Video_inpainting_DPO/finetune-stage2/converted_weights_step48000"),
            Path("/mnt/nas/hj/H20_Video_inpainting_DPO/finetune-stage2/converted_weights_step34000"),
            Path("/home/nvme01/H20_Video_inpainting_DPO/finetune-stage2/converted_weights_step48000"),
            Path("/home/nvme01/H20_Video_inpainting_DPO/finetune-stage2/converted_weights_step34000"),
        ]
    )
    candidates.extend(
        glob_many(
            [
                "/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/sft/stage2/*/last_weights",
                "/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/sft/stage2/*/converted_weights",
                "/home/nvme01/H20_Video_inpainting_DPO/experiments/sft/stage2/*/last_weights",
                "/home/nvme01/H20_Video_inpainting_DPO/experiments/sft/stage2/*/converted_weights",
            ]
        )
    )
    return dedupe(candidates)


def md_path_list(title: str, reports: Sequence[WeightDirReport]) -> List[str]:
    lines = [f"## {title}", "", "| label | exists | class | path | note |", "| --- | --- | --- | --- | --- |"]
    for report in reports:
        cfg = report.configs.get("unet_main/config.json", {})
        cls = cfg.get("_class_name", "")
        lines.append(f"| `{report.label}` | {report.exists} | `{cls}` | `{report.path}` | {report.note} |")
    lines.append("")
    return lines


def key_sample_block(report: WeightDirReport) -> List[str]:
    lines = [f"### {report.label}", "", f"Path: `{report.path}`", ""]
    if not report.exists:
        lines.append("Status: missing")
        lines.append("")
        return lines
    lines.append("Files:")
    lines.append("")
    for rel in report.files[:40]:
        lines.append(f"- `{rel}`")
    lines.append("")
    for rel, cfg in report.configs.items():
        lines.append(f"- `{rel}` class: `{cfg.get('_class_name', '')}`")
    lines.append("")
    lines.append(f"UNet key count: `{report.unet_key_count}`")
    lines.append("")
    lines.append("Spatial key sample:")
    for key in report.spatial_key_sample[:20]:
        lines.append(f"- `{key}`")
    lines.append("")
    lines.append("Temporal/motion key sample:")
    for key in report.temporal_key_sample[:20]:
        lines.append(f"- `{key}`")
    lines.append("")
    if report.uncertain_key_sample:
        lines.append("Uncertain/non-motion-preserved key sample:")
        for key in report.uncertain_key_sample[:20]:
            lines.append(f"- `{key}`")
        lines.append("")
    return lines


def write_markdown(
    path: Path,
    stage1_reports: Sequence[WeightDirReport],
    dpo_stage2_reports: Sequence[WeightDirReport],
    sft_stage2_reports: Sequence[WeightDirReport],
) -> None:
    lines = [
        "# DiffuEraser Stage Checkpoint Structure",
        "",
        "Purpose: audit whether DPO Stage1 spatial/appearance weights can be combined with frozen SFT Stage2 temporal/motion weights.",
        "",
        "## Interpretation",
        "",
        "- Stage1 exported weights normally contain `unet_main/` as `UNet2DConditionModel` plus `brushnet/`.",
        "- Stage2 exported weights normally contain `unet_main/` as `UNetMotionModel` plus `brushnet/`.",
        "- Spatial / appearance modules are copied from DPO Stage1: `conv_in`, `time_proj`, `time_embedding`, `down_blocks`, `up_blocks`, `mid_block`, `conv_norm_out`, `conv_act`, `conv_out`, and `brushnet`.",
        "- Temporal / motion modules are preserved from SFT Stage2: keys containing `motion`, `temporal`, `motion_modules`, `temporal_transformer`, or `temporal_attention`.",
        "- Current full-mask/partial-mask eval loaders take a single `weights_path`; they cannot directly pass separate `stage1_weights_dir` and `motion_weights_dir` without a wrapper or physical hybrid export.",
        "- The existing DiffuEraser Stage2 code already performs the safe operation: load a motion UNet, copy Stage1 2D modules into it, and load BrushNet from Stage1.",
        "",
    ]
    lines.extend(md_path_list("DPO Stage1 Candidates", stage1_reports))
    lines.extend(md_path_list("DPO Stage2 Candidates", dpo_stage2_reports))
    lines.extend(md_path_list("SFT / Base Stage2 Candidates", sft_stage2_reports))
    lines.append("## Detailed Samples")
    lines.append("")
    for report in list(stage1_reports) + list(dpo_stage2_reports) + list(sft_stage2_reports):
        lines.extend(key_sample_block(report))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", action="append", type=Path, default=[])
    parser.add_argument("--dpo_stage2", action="append", type=Path, default=[])
    parser.add_argument("--sft_stage2", action="append", type=Path, default=[])
    parser.add_argument("--report_path", type=Path, default=Path("reports/diffueraser_stage_checkpoint_structure.md"))
    parser.add_argument("--json_path", type=Path, default=None)
    args = parser.parse_args()

    stage1_paths = dedupe(args.stage1 or default_stage1_candidates())
    dpo_stage2_paths = dedupe(args.dpo_stage2 or default_stage2_candidates())
    sft_stage2_paths = dedupe(args.sft_stage2 or default_sft_stage2_candidates())

    stage1_reports = [inspect_dir(f"DPO_Stage1_{i}", path) for i, path in enumerate(stage1_paths, start=1)]
    dpo_stage2_reports = [inspect_dir(f"DPO_Stage2_{i}", path) for i, path in enumerate(dpo_stage2_paths, start=1)]
    sft_stage2_reports = [inspect_dir(f"SFT_or_base_Stage2_{i}", path) for i, path in enumerate(sft_stage2_paths, start=1)]

    write_markdown(args.report_path, stage1_reports, dpo_stage2_reports, sft_stage2_reports)
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "stage1": [r.__dict__ for r in stage1_reports],
            "dpo_stage2": [r.__dict__ for r in dpo_stage2_reports],
            "sft_stage2": [r.__dict__ for r in sft_stage2_reports],
        }
        args.json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"[inspect-stage-weights] report={args.report_path}")
    if args.json_path:
        print(f"[inspect-stage-weights] json={args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
