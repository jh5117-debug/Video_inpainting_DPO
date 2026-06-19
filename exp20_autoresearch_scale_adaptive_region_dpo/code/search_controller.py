"""Safe best-first search controller for Exp20.

The controller only generates immutable configs and appends result rows. It does
not edit Python, reset git, or change evaluation code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


RESULT_FIELDS = [
    "trial_id",
    "parent_id",
    "config_hash",
    "branch_commit",
    "radius_mode",
    "radius_value",
    "adaptive_k",
    "boundary_contribution",
    "aggregation",
    "seed",
    "gpu_ids",
    "world_size",
    "effective_batch",
    "train_seconds",
    "num_steps",
    "peak_vram",
    "dev_psnr",
    "dev_ssim",
    "dev_lpips",
    "dev_vfid_or_fvd",
    "dev_tc",
    "dev_ewarp",
    "dev_mask_psnr",
    "dev_boundary_psnr",
    "implicit_acc",
    "loser_dominant_ratio",
    "max_grad_norm",
    "status",
    "keep_reason",
    "description",
    "checkpoint_path",
    "log_path",
]


@dataclass(frozen=True)
class TrialConfig:
    trial_id: str
    parent_id: str
    radius_mode: str
    radius_value: float
    boundary_contribution: float
    aggregation: str
    adaptive_k: float = 0.0
    seed: int = 20260619
    stage: str = "stage1"
    train_minutes: int = 30
    description: str = ""
    base_checkpoint_identity: str = ""
    dataset_manifest_hash: str = ""
    training_code_hash: str = ""
    evaluator_hash: str = ""
    beta: float = 10.0
    gap_eps: float = 1e-8
    lose_gap_weight: float = 0.25
    lose_gap_clip_tau: float = 1.0
    winner_abs_reg_weight: float = 0.05
    winner_gap_reg_weight: float = 1.0
    winner_gap_reg_margin: float = 0.0
    mask_contribution: float = 1.0
    outside_contribution: float = 0.05
    checkpoint_path: str = ""
    log_path: str = ""
    gpu_ids: str = ""

    def stable_payload(self) -> Dict[str, object]:
        payload = asdict(self)
        for key in (
            "trial_id",
            "parent_id",
            "description",
            "checkpoint_path",
            "log_path",
            "gpu_ids",
        ):
            payload.pop(key, None)
        return payload

    def config_hash(self) -> str:
        raw = json.dumps(self.stable_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]


class SearchController:
    def __init__(self, exp_dir: Path):
        self.exp_dir = exp_dir
        self.config_dir = exp_dir / "configs"
        self.queue_dir = exp_dir / "queue"
        self.results_path = exp_dir / "results.tsv"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    def ensure_results_header(self) -> None:
        if self.results_path.exists() and self.results_path.stat().st_size > 0:
            return
        with self.results_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, delimiter="\t")
            writer.writeheader()

    def existing_hashes(self) -> set[str]:
        self.ensure_results_header()
        hashes = set()
        for path in self.config_dir.glob("*.json"):
            try:
                hashes.add(json.loads(path.read_text()).get("config_hash", ""))
            except Exception:
                continue
        return hashes

    def enqueue(self, cfg: TrialConfig) -> Optional[Path]:
        h = cfg.config_hash()
        if h in self.existing_hashes():
            return None
        payload = asdict(cfg)
        payload["config_hash"] = h
        path = self.config_dir / f"{cfg.trial_id}_{h}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        queue_path = self.queue_dir / path.name
        queue_path.write_text(str(path) + "\n")
        return path

    def initialize_roots(self, search_space: Dict[str, object]) -> List[Path]:
        out = []
        for i, node in enumerate(search_space.get("root_nodes", []), start=1):
            cfg = TrialConfig(
                trial_id=f"root_{i:03d}",
                parent_id="",
                radius_mode=str(node["radius_mode"]),
                radius_value=float(node.get("radius_value", 0.0)),
                boundary_contribution=float(node["boundary_contribution"]),
                aggregation=str(node["aggregation"]),
                adaptive_k=float(node.get("adaptive_k", 0.0)),
                description=str(node.get("description", "root")),
            )
            path = self.enqueue(cfg)
            if path is not None:
                out.append(path)
        return out


def load_search_space(path: Path) -> Dict[str, object]:
    text = path.read_text()
    if yaml is None:
        raise RuntimeError("PyYAML is required to read search_space.yaml")
    return yaml.safe_load(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", default="exp20_autoresearch_scale_adaptive_region_dpo")
    parser.add_argument("--search-space", default="exp20_autoresearch_scale_adaptive_region_dpo/search_space.yaml")
    parser.add_argument("--init-roots", action="store_true")
    args = parser.parse_args()
    controller = SearchController(Path(args.exp_dir))
    controller.ensure_results_header()
    if args.init_roots:
        paths = controller.initialize_roots(load_search_space(Path(args.search_space)))
        print(f"enqueued_roots={len(paths)}")
        for path in paths:
            print(path)


if __name__ == "__main__":
    main()
