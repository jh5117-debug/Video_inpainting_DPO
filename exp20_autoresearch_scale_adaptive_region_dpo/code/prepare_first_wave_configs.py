#!/usr/bin/env python3
"""Prepare immutable P0-P5 first-wave configs in an isolated queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp20_autoresearch_scale_adaptive_region_dpo.code.search_controller import TrialConfig, load_search_space  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-space", default="exp20_autoresearch_scale_adaptive_region_dpo/search_space.yaml")
    parser.add_argument("--output-root", default="exp20_autoresearch_scale_adaptive_region_dpo/first_wave")
    parser.add_argument("--dev-manifest", default="exp20_autoresearch_scale_adaptive_region_dpo/manifests/dev_boundary_search_v1.jsonl")
    args = parser.parse_args()

    out_root = Path(args.output_root)
    config_dir = out_root / "configs"
    queue_dir = out_root / "queue"
    config_dir.mkdir(parents=True, exist_ok=True)
    queue_dir.mkdir(parents=True, exist_ok=True)
    space = load_search_space(Path(args.search_space))
    manifest_hash = sha256_file(Path(args.dev_manifest))
    training_hash = sha256_file(Path("exp20_autoresearch_scale_adaptive_region_dpo/code/train_exp20_stage1.py"))
    evaluator_hash = sha256_file(Path("exp20_autoresearch_scale_adaptive_region_dpo/code/evaluate_trial.py"))
    branch_commit = ""
    try:
        import subprocess

        branch_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        pass

    written = []
    seen: set[str] = set()
    for node in space.get("root_nodes", []):
        cfg = TrialConfig(
            trial_id=str(node["trial_id"]),
            parent_id="",
            radius_mode=str(node["radius_mode"]),
            radius_value=float(node.get("radius_value", 0.0)),
            boundary_contribution=float(node["boundary_contribution"]),
            aggregation=str(node["aggregation"]),
            adaptive_k=float(node.get("adaptive_k", 0.0)),
            description=str(node.get("description", "")),
            base_checkpoint_identity="/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000",
            dataset_manifest_hash=manifest_hash,
            training_code_hash=training_hash,
            evaluator_hash=evaluator_hash,
        )
        h = cfg.config_hash()
        if h in seen:
            continue
        seen.add(h)
        payload = cfg.__dict__.copy()
        payload["config_hash"] = h
        payload["branch_commit"] = branch_commit
        path = config_dir / f"{cfg.trial_id}_{h}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        (queue_dir / path.name).write_text(str(path) + "\n")
        written.append(path)
    manifest = {
        "status": "FIRST_WAVE_CONFIGS_PREPARED",
        "count": len(written),
        "config_dir": str(config_dir),
        "queue_dir": str(queue_dir),
        "dev_manifest": args.dev_manifest,
        "dev_manifest_sha256": manifest_hash,
        "configs": [str(p) for p in written],
    }
    (out_root / "trial_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
