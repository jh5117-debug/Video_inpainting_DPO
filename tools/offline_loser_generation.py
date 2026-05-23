from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


MODEL_ENV = {
    "diffueraser": "DIFFUERASER_WEIGHT_ROOT",
    "propainter": "PROPAINTER_WEIGHT_ROOT",
    "cococo": "COCOCO_WEIGHT_ROOT",
    "minimax_remover": "MINIMAX_REMOVER_WEIGHT_ROOT",
}

MANIFEST_FIELDS = [
    "sample_id",
    "source_video_id",
    "mask_id",
    "prompt",
    "win_video_path",
    "raw_loser_video_path",
    "comp_loser_video_path",
    "final_loser_video_path",
    "mask_path",
    "mask_mode",
    "mask_convention",
    "comp",
    "generation_model",
    "source_dataset",
    "seed",
    "fps",
    "num_frames",
    "height",
    "width",
    "mask_area_ratio",
    "mask_bbox",
    "status",
]


@dataclass
class GenerationPlan:
    source_dataset: str
    output_root: str
    model_name: str
    mask_mode: str
    mask_convention: str
    comp: bool
    offline: bool
    num_samples: int | None
    num_masks_per_video: int
    skip_existing: bool
    resume: bool
    limit: int | None
    start_index: int | None
    end_index: int | None
    seed: int
    save_manifest: str | None
    weight_env: str
    weight_root: str | None


def _str_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def _canonical_model_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    if normalized == "cocococ":
        normalized = "cococo"
    if normalized == "minimax":
        normalized = "minimax_remover"
    if normalized == "all":
        return normalized
    if normalized not in MODEL_ENV:
        allowed = ", ".join(sorted([*MODEL_ENV, "all"]))
        raise argparse.ArgumentTypeError(f"unknown model {name!r}; expected one of: {allowed}")
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan offline loser generation for VideoDPO/DiffuEraser data ablations."
    )
    parser.add_argument("--source_dataset", required=True, help="Source preference dataset root or manifest.")
    parser.add_argument("--output_root", required=True, help="Directory for generated losers and manifests.")
    parser.add_argument(
        "--model_name",
        required=True,
        type=_canonical_model_name,
        help="diffueraser, propainter, cococo, cocococ, minimax, minimax_remover, or all.",
    )
    parser.add_argument("--mask_mode", choices=["full", "partial"], required=True)
    parser.add_argument(
        "--mask_convention",
        default="unconfirmed",
        help=(
            "Human-readable mask convention for this generation run, e.g. "
            "'diffueraser_internal_0_hole_pil_white_hole'."
        ),
    )
    parser.add_argument("--comp", type=_str_bool, default=False)
    parser.add_argument("--offline", type=_str_bool, default=True)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--num_masks_per_video", type=int, default=1)
    parser.add_argument("--skip_existing", type=_str_bool, default=True)
    parser.add_argument("--resume", type=_str_bool, default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=None)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_manifest", default=None)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate paths and print the plan without invoking model inference.",
    )
    parser.add_argument(
        "--allow_missing_assets",
        action="store_true",
        help="Allow missing source/weight paths during dry-run planning only.",
    )
    return parser


def _require_path(path: str | None, label: str, allow_missing: bool) -> None:
    if not path:
        if allow_missing:
            return
        raise SystemExit(f"[error] missing {label}")
    if not Path(path).exists() and not allow_missing:
        raise SystemExit(f"[error] {label} does not exist: {path}")


def make_plan(args: argparse.Namespace) -> GenerationPlan:
    weight_env = "MULTI_MODEL" if args.model_name == "all" else MODEL_ENV[args.model_name]
    weight_root = None if args.model_name == "all" else os.environ.get(weight_env)
    allow_missing = bool(args.dry_run and args.allow_missing_assets)
    _require_path(args.source_dataset, "source_dataset", allow_missing)
    if args.model_name != "all":
        _require_path(weight_root, weight_env, allow_missing)
    if args.num_masks_per_video < 1:
        raise SystemExit("[error] --num_masks_per_video must be >= 1")
    if args.mask_mode == "full" and args.num_masks_per_video != 1:
        raise SystemExit("[error] full-mask generation should use --num_masks_per_video 1")
    return GenerationPlan(
        source_dataset=args.source_dataset,
        output_root=args.output_root,
        model_name=args.model_name,
        mask_mode=args.mask_mode,
        mask_convention=args.mask_convention,
        comp=bool(args.comp),
        offline=bool(args.offline),
        num_samples=args.num_samples,
        num_masks_per_video=args.num_masks_per_video,
        skip_existing=bool(args.skip_existing),
        resume=bool(args.resume),
        limit=args.limit,
        start_index=args.start_index,
        end_index=args.end_index,
        seed=args.seed,
        save_manifest=args.save_manifest,
        weight_env=weight_env,
        weight_root=weight_root,
    )


def write_manifest_schema(path: str, plan: GenerationPlan) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "fields": MANIFEST_FIELDS,
        "generation_plan": asdict(plan),
        "items": [],
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.offline:
        raise SystemExit("[error] online loser generation is documented as future work; use --offline true.")
    if args.comp and args.mask_mode != "partial":
        raise SystemExit("[error] --comp true is only meaningful with --mask_mode partial.")
    plan = make_plan(args)
    print(json.dumps(asdict(plan), indent=2))
    if args.save_manifest and args.dry_run:
        write_manifest_schema(args.save_manifest, plan)
        print(f"[dry-run] wrote manifest schema: {args.save_manifest}")
    if args.dry_run:
        return 0
    raise SystemExit(
        "[error] generation dispatch is intentionally not implemented in this scaffold. "
        "Wire this plan to DPO_finetune/infer_*_candidate.py after confirming the target model env/weights."
    )


if __name__ == "__main__":
    raise SystemExit(main())
