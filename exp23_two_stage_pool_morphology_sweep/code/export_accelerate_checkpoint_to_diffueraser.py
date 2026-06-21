#!/usr/bin/env python3
"""Export an Exp23 accelerate checkpoint as a DiffuEraser weights root.

Exp23 trainers save intermediate checkpoints with accelerate as:

    checkpoint-N/model.safetensors
    checkpoint-N/model_1.safetensors

For the current Stage1/Stage2 trainers, ``model`` is ``unet_main`` and
``model_1`` is ``brushnet``. The canonical DAVIS evaluator expects a
DiffuEraser root containing ``unet_main/config.json`` plus
``unet_main/diffusion_pytorch_model.safetensors`` and the same for
``brushnet``. This utility performs only that packaging step and writes an
identity manifest for audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

from safetensors.torch import load_file


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def key_summary(path: Path) -> Dict[str, object]:
    state = load_file(str(path), device="cpu")
    keys = sorted(state.keys())
    return {
        "num_keys": len(keys),
        "first_keys": keys[:12],
        "sha256": sha256_file(path),
    }


def copy_config_files(template_subdir: Path, out_subdir: Path) -> List[str]:
    copied: List[str] = []
    out_subdir.mkdir(parents=True, exist_ok=True)
    for path in sorted(template_subdir.iterdir()):
        if path.is_file() and path.name != "diffusion_pytorch_model.safetensors":
            shutil.copy2(path, out_subdir / path.name)
            copied.append(path.name)
    return copied


def assert_same_keys(left: Path, right: Path, label: str) -> None:
    left_keys = set(load_file(str(left), device="cpu").keys())
    right_keys = set(load_file(str(right), device="cpu").keys())
    if left_keys != right_keys:
        missing = sorted(right_keys - left_keys)[:20]
        unexpected = sorted(left_keys - right_keys)[:20]
        raise RuntimeError(
            f"{label} key mismatch: missing={missing} unexpected={unexpected} "
            f"left={left} right={right}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True, type=Path)
    parser.add_argument("--template_weights", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--model_label", required=True)
    parser.add_argument("--stage", required=True, choices=["stage1", "stage2"])
    parser.add_argument("--step", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate_template_keys", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir
    template = args.template_weights
    output = args.output_dir

    unet_src = checkpoint_dir / "model.safetensors"
    brush_src = checkpoint_dir / "model_1.safetensors"
    require_file(unet_src, "accelerate unet model")
    require_file(brush_src, "accelerate brushnet model")
    require_file(template / "unet_main" / "config.json", "template unet config")
    require_file(template / "brushnet" / "config.json", "template brushnet config")

    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite non-empty output_dir: {output}")
        shutil.rmtree(output)

    if args.validate_template_keys:
        template_unet = template / "unet_main" / "diffusion_pytorch_model.safetensors"
        template_brush = template / "brushnet" / "diffusion_pytorch_model.safetensors"
        require_file(template_unet, "template unet weights")
        require_file(template_brush, "template brushnet weights")
        assert_same_keys(unet_src, template_unet, "unet_main")
        assert_same_keys(brush_src, template_brush, "brushnet")

    unet_out = output / "unet_main"
    brush_out = output / "brushnet"
    copied_unet = copy_config_files(template / "unet_main", unet_out)
    copied_brush = copy_config_files(template / "brushnet", brush_out)
    shutil.copy2(unet_src, unet_out / "diffusion_pytorch_model.safetensors")
    shutil.copy2(brush_src, brush_out / "diffusion_pytorch_model.safetensors")

    manifest = {
        "checkpoint_kind": "exp23_accelerate_checkpoint_export",
        "model_label": args.model_label,
        "stage": args.stage,
        "step": args.step,
        "checkpoint_dir": str(checkpoint_dir),
        "template_weights": str(template),
        "output_dir": str(output),
        "mapping": {
            "model.safetensors": "unet_main/diffusion_pytorch_model.safetensors",
            "model_1.safetensors": "brushnet/diffusion_pytorch_model.safetensors",
        },
        "copied_config_files": {
            "unet_main": copied_unet,
            "brushnet": copied_brush,
        },
        "unet_main": key_summary(unet_src),
        "brushnet": key_summary(brush_src),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
