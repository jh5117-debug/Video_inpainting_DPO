#!/usr/bin/env python
"""Build a DiffuEraser hybrid checkpoint.

Hybrid policy:

    spatial / appearance weights  <- DPO Stage1
    temporal / motion weights     <- SFT or base Stage2

This mirrors the vetted Stage2 loader pattern already used in training: load a
motion UNet first, then copy Stage1 2D modules into the motion UNet while
leaving motion-specific modules untouched.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEMPORAL_TOKENS = (
    "motion",
    "temporal",
    "time_mixer",
    "motion_modules",
    "temporal_transformer",
    "temporal_attention",
)


@dataclass
class CopyReport:
    loaded_from_dpo_stage1: List[str] = field(default_factory=list)
    loaded_from_sft_stage2: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    shape_mismatch: List[Dict[str, Any]] = field(default_factory=list)
    unexpected: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    uncertain_preserved_from_sft_stage2: List[str] = field(default_factory=list)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}")


def require_exported_weights(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    for sub in ["unet_main", "brushnet"]:
        if not (path / sub).exists():
            raise FileNotFoundError(f"{label} is not an exported DiffuEraser weights dir; missing {sub}/ in {path}")


def is_temporal_key(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in TEMPORAL_TOKENS)


def copy_module_state(src, dst, prefix: str, report: CopyReport, dry_run: bool) -> None:
    src_state = src.state_dict()
    dst_state = dst.state_dict()
    updates = {}
    for key, tensor in src_state.items():
        full_key = f"{prefix}.{key}" if key else prefix
        if key not in dst_state:
            report.missing.append(full_key)
            continue
        if tuple(dst_state[key].shape) != tuple(tensor.shape):
            report.shape_mismatch.append(
                {
                    "key": full_key,
                    "dpo_stage1_shape": list(tensor.shape),
                    "sft_stage2_shape": list(dst_state[key].shape),
                }
            )
            continue
        updates[key] = tensor
        report.loaded_from_dpo_stage1.append(full_key)
    for key in dst_state.keys():
        if key not in src_state:
            full_key = f"{prefix}.{key}" if key else prefix
            report.unexpected.append(full_key)
    if updates and not dry_run:
        merged = dict(dst_state)
        merged.update(updates)
        dst.load_state_dict(merged, strict=False)


def copy_if_present(src_parent, dst_parent, attr: str, prefix: str, report: CopyReport, dry_run: bool) -> None:
    if not hasattr(src_parent, attr) or not hasattr(dst_parent, attr):
        report.skipped.append(f"{prefix}: attribute missing on source or destination")
        return
    src = getattr(src_parent, attr)
    dst = getattr(dst_parent, attr)
    if src is None or dst is None:
        report.skipped.append(f"{prefix}: source or destination is None")
        return
    copy_module_state(src, dst, prefix, report, dry_run=dry_run)


def copy_unet2d_to_motion(src, dst, report: CopyReport, dry_run: bool) -> None:
    copy_if_present(src, dst, "conv_in", "unet_main.conv_in", report, dry_run)
    copy_if_present(src, dst, "time_proj", "unet_main.time_proj", report, dry_run)
    copy_if_present(src, dst, "time_embedding", "unet_main.time_embedding", report, dry_run)

    for i, src_block in enumerate(src.down_blocks):
        if i >= len(dst.down_blocks):
            report.skipped.append(f"unet_main.down_blocks.{i}: destination block missing")
            continue
        dst_block = dst.down_blocks[i]
        copy_if_present(src_block, dst_block, "resnets", f"unet_main.down_blocks.{i}.resnets", report, dry_run)
        copy_if_present(src_block, dst_block, "attentions", f"unet_main.down_blocks.{i}.attentions", report, dry_run)
        if getattr(src_block, "downsamplers", None) and getattr(dst_block, "downsamplers", None):
            copy_module_state(
                src_block.downsamplers,
                dst_block.downsamplers,
                f"unet_main.down_blocks.{i}.downsamplers",
                report,
                dry_run=dry_run,
            )

    for i, src_block in enumerate(src.up_blocks):
        if i >= len(dst.up_blocks):
            report.skipped.append(f"unet_main.up_blocks.{i}: destination block missing")
            continue
        dst_block = dst.up_blocks[i]
        copy_if_present(src_block, dst_block, "resnets", f"unet_main.up_blocks.{i}.resnets", report, dry_run)
        copy_if_present(src_block, dst_block, "attentions", f"unet_main.up_blocks.{i}.attentions", report, dry_run)
        if getattr(src_block, "upsamplers", None) and getattr(dst_block, "upsamplers", None):
            copy_module_state(
                src_block.upsamplers,
                dst_block.upsamplers,
                f"unet_main.up_blocks.{i}.upsamplers",
                report,
                dry_run=dry_run,
            )

    copy_if_present(src.mid_block, dst.mid_block, "resnets", "unet_main.mid_block.resnets", report, dry_run)
    copy_if_present(src.mid_block, dst.mid_block, "attentions", "unet_main.mid_block.attentions", report, dry_run)
    copy_if_present(src, dst, "conv_norm_out", "unet_main.conv_norm_out", report, dry_run)
    copy_if_present(src, dst, "conv_act", "unet_main.conv_act", report, dry_run)
    copy_if_present(src, dst, "conv_out", "unet_main.conv_out", report, dry_run)


def summarize_preserved_motion(motion_unet, report: CopyReport) -> None:
    copied = set(report.loaded_from_dpo_stage1)
    for key in motion_unet.state_dict().keys():
        full_key = f"unet_main.{key}"
        if full_key in copied:
            continue
        if is_temporal_key(full_key):
            report.loaded_from_sft_stage2.append(full_key)
        else:
            report.uncertain_preserved_from_sft_stage2.append(full_key)


def copy_brushnet(dpo_stage1: Path, output_last_weights: Path, dry_run: bool, report: CopyReport) -> None:
    src = dpo_stage1 / "brushnet"
    dst = output_last_weights / "brushnet"
    if dry_run:
        for path in sorted(p for p in src.rglob("*") if p.is_file()):
            report.loaded_from_dpo_stage1.append(f"brushnet/{path.relative_to(src).as_posix()}")
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    for path in sorted(p for p in dst.rglob("*") if p.is_file()):
        report.loaded_from_dpo_stage1.append(f"brushnet/{path.relative_to(dst).as_posix()}")


def write_reports(
    output_dir: Path,
    report_path: Path,
    report: CopyReport,
    args: argparse.Namespace,
    dry_run: bool,
) -> None:
    payload = {
        "mode": args.mode,
        "dry_run": dry_run,
        "dpo_stage1_weights": str(args.dpo_stage1_weights),
        "sft_stage2_weights": str(args.sft_stage2_weights),
        "output_dir": str(args.output_dir),
        "counts": {
            "loaded_from_dpo_stage1": len(report.loaded_from_dpo_stage1),
            "loaded_from_sft_stage2": len(report.loaded_from_sft_stage2),
            "skipped": len(report.skipped),
            "shape_mismatch": len(report.shape_mismatch),
            "unexpected": len(report.unexpected),
            "missing": len(report.missing),
            "uncertain_preserved_from_sft_stage2": len(report.uncertain_preserved_from_sft_stage2),
        },
        "loaded_from_dpo_stage1": report.loaded_from_dpo_stage1,
        "loaded_from_sft_stage2": report.loaded_from_sft_stage2,
        "skipped": report.skipped,
        "shape_mismatch": report.shape_mismatch,
        "unexpected": report.unexpected,
        "missing": report.missing,
        "uncertain_preserved_from_sft_stage2": report.uncertain_preserved_from_sft_stage2,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "key_merge_report.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# DPO-S1 + SFT-S2 Hybrid Key Merge Report",
        "",
        f"mode: `{args.mode}`",
        f"dry_run: `{dry_run}`",
        f"dpo_stage1_weights: `{args.dpo_stage1_weights}`",
        f"sft_stage2_weights: `{args.sft_stage2_weights}`",
        f"output_dir: `{args.output_dir}`",
        "",
        "## Counts",
        "",
        "| category | count |",
        "| --- | ---: |",
    ]
    for key, value in payload["counts"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Safety Notes",
            "",
            "- DPO Stage1 spatial/appearance modules are copied into the Stage2 motion UNet.",
            "- SFT Stage2 motion/temporal keys are preserved in the motion UNet.",
            "- DPO Stage1 BrushNet is copied as the hybrid BrushNet.",
            "- Uncertain preserved keys are left from SFT Stage2 and listed for audit; they are not silently overwritten by SFT over DPO spatial modules.",
            "",
        ]
    )
    for title, values in [
        ("Loaded From DPO Stage1 Sample", report.loaded_from_dpo_stage1),
        ("Loaded From SFT Stage2 Motion Sample", report.loaded_from_sft_stage2),
        ("Uncertain Preserved From SFT Stage2 Sample", report.uncertain_preserved_from_sft_stage2),
        ("Shape Mismatch", [json.dumps(x, sort_keys=True) for x in report.shape_mismatch]),
        ("Missing", report.missing),
        ("Skipped", report.skipped),
    ]:
        lines.extend([f"## {title}", ""])
        if values:
            for value in values[:80]:
                lines.append(f"- `{value}`")
            if len(values) > 80:
                lines.append(f"- ... {len(values) - 80} more")
        else:
            lines.append("- none")
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = "\n".join(lines).rstrip() + "\n"
    report_path.write_text(report_text, encoding="utf-8")
    default_md_path = output_dir / "key_merge_report.md"
    if default_md_path.resolve() != report_path.resolve():
        default_md_path.write_text(report_text, encoding="utf-8")

    manifest = {
        "checkpoint_kind": "dpoS1_sftS2_hybrid",
        "mode": args.mode,
        "dpo_stage1_weights": str(args.dpo_stage1_weights),
        "sft_stage2_weights": str(args.sft_stage2_weights),
        "last_weights": str(args.output_dir / "last_weights"),
        "key_merge_report_json": str(json_path),
        "key_merge_report_md": str(report_path),
        "dry_run": dry_run,
    }
    (output_dir / "hybrid_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpo_stage1_weights", required=True, type=Path)
    parser.add_argument("--sft_stage2_weights", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--mode", default="dpo_spatial_sft_motion", choices=["dpo_spatial_sft_motion"])
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--strict", type=parse_bool, default=False)
    parser.add_argument("--report_path", type=Path, default=None)
    parser.add_argument("--variant", default=None)
    args = parser.parse_args()

    require_exported_weights(args.dpo_stage1_weights, "DPO Stage1 weights")
    require_exported_weights(args.sft_stage2_weights, "SFT Stage2 weights")

    output_dir = args.output_dir
    output_last_weights = output_dir / "last_weights"
    report_path = args.report_path or output_dir / "key_merge_report.md"
    if not args.dry_run and output_last_weights.exists() and any(output_last_weights.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing hybrid last_weights: {output_last_weights}")

    from libs.brushnet_CA import BrushNetModel  # noqa: F401 - import validates local class availability
    from libs.unet_2d_condition import UNet2DConditionModel
    from libs.unet_motion_model import UNetMotionModel

    report = CopyReport()
    print(f"[hybrid-builder] loading SFT Stage2 motion UNet: {args.sft_stage2_weights}")
    motion_unet = UNetMotionModel.from_pretrained(str(args.sft_stage2_weights), subfolder="unet_main")
    print(f"[hybrid-builder] loading DPO Stage1 spatial UNet: {args.dpo_stage1_weights}")
    stage1_unet = UNet2DConditionModel.from_pretrained(
        str(args.dpo_stage1_weights), subfolder="unet_main", variant=args.variant
    )

    copy_unet2d_to_motion(stage1_unet, motion_unet, report, dry_run=args.dry_run)
    summarize_preserved_motion(motion_unet, report)
    copy_brushnet(args.dpo_stage1_weights, output_last_weights, args.dry_run, report)

    if args.strict and (report.shape_mismatch or report.missing):
        write_reports(output_dir, report_path, report, args, dry_run=args.dry_run)
        raise RuntimeError(
            f"Strict hybrid build failed: shape_mismatch={len(report.shape_mismatch)} missing={len(report.missing)}"
        )

    if not args.dry_run:
        output_last_weights.mkdir(parents=True, exist_ok=True)
        motion_unet.save_pretrained(output_last_weights / "unet_main")

    write_reports(output_dir, report_path, report, args, dry_run=args.dry_run)
    print(f"[hybrid-builder] output={output_dir}")
    print(f"[hybrid-builder] report={report_path}")
    if args.dry_run:
        print("[hybrid-builder] dry-run only; no weights were written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
