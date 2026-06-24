#!/usr/bin/env python3
"""Three-lane continuation controller for Exp25/26/27.

This controller is deliberately append-only and milestone-oriented. It assumes
GPU cleanup has already been done by the caller, then assigns one selected GPU
to each lane:

- Lane A: Exp25 Gate32 yield review and too-close seed2 supplementation.
- Lane B: Exp26 Probe4 review, then Gate16 materialize/mask/infer/review.
- Lane C: Exp27 SDPO/Linear/LocalDPO parity and smoke gates.

It does not start training, Gate128 expansion, Gate64 expansion, or RC-FPO.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_jsonl(path: Path, limit: int = 0) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def append_csv(path: Path, row: dict[str, Any], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def latest_snapshot(root: Path, prefix: str) -> Path:
    matches = sorted([p for p in root.glob(f"{prefix}_*") if p.is_dir() and (p / ".snapshot_HEAD").is_file()], key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No snapshot for {prefix!r} under {root}")
    return matches[-1]


class Controller:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_root: Path = args.run_root
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_root / "three_lane_state.json"
        self.lock = threading.Lock()
        self.stop = False
        self.running: dict[str, dict[str, Any]] = {}
        self.exp25 = latest_snapshot(args.snapshot_root, "exp25")
        self.exp26 = latest_snapshot(args.snapshot_root, "exp26")
        self.exp27 = latest_snapshot(args.snapshot_root, "exp27")

    def mark(self, task: str, status: str, **extra: Any) -> None:
        with self.lock:
            state = read_json(self.state_path, {"tasks": {}, "running": {}})
            item = state.setdefault("tasks", {}).setdefault(task, {})
            item.update({"status": status, "updated_at": utc_now(), **extra})
            state["running"] = self.running
            write_json(self.state_path, state)

    def run_logged(self, task: str, cmd: list[str], cwd: Path, gpu: int | None = None, env_extra: dict[str, str] | None = None, timeout: int | None = None) -> int:
        log = self.run_root / f"{task}.log"
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.update(env_extra or {})
        self.mark(task, "running", gpu=gpu, command=" ".join(cmd), log_path=str(log))
        with log.open("a", encoding="utf-8") as f:
            f.write(f"\n[start {utc_now()}] gpu={gpu}\n[cmd] {' '.join(cmd)}\n[cwd] {cwd}\n")
            f.flush()
            proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            self.running[task] = {"pid": proc.pid, "pgid": os.getpgid(proc.pid), "gpu": gpu, "log_path": str(log), "command": cmd, "started_at": utc_now()}
            self.mark(task, "running", pid=proc.pid, pgid=os.getpgid(proc.pid), gpu=gpu, log_path=str(log))
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                time.sleep(10)
                if proc.poll() is None:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                rc = proc.wait()
                f.write(f"\n[timeout] task killed after {timeout}s rc={rc}\n")
            self.running.pop(task, None)
            f.write(f"\n[end {utc_now()}] rc={rc}\n")
        self.mark(task, "completed" if rc == 0 else "failed", returncode=rc, gpu=gpu, log_path=str(log))
        return rc

    def lane_a_exp25(self, gpu: int) -> None:
        try:
            manifest = Path("/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/materialized/gate32_canonical_d0_24f/gate32_materialized.jsonl")
            candidate_root = Path("/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/preference_candidates/gate32_canonical_raw6_d0")
            review = self.run_root / "exp25_gate32_yield_review"
            rc = self.run_logged(
                "exp25_gate32_yield_review",
                [
                    sys.executable,
                    str(self.exp25 / "exp25_vor_or_preference_data" / "scripts" / "analyze_gate32_yield.py"),
                    "--manifest",
                    str(manifest),
                    "--candidate-root",
                    str(candidate_root),
                    "--output-dir",
                    str(review),
                    "--limit",
                    "32",
                ],
                self.exp25,
                gpu=None,
            )
            if rc != 0:
                return
            summary = read_json(review / "gate32_yield_summary.json", {})
            too_close = int(summary.get("counts", {}).get("too-close", 0))
            if too_close > 0:
                out_seed2 = Path("/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/preference_candidates/gate32_canonical_raw6_seed2_too_close")
                self.run_logged(
                    "exp25_gate32_too_close_seed2",
                    [
                        sys.executable,
                        str(self.exp25 / "exp25_vor_or_preference_data" / "scripts" / "run_vor_or_model_smoke.py"),
                        "--model",
                        "diffueraser",
                        "--manifest",
                        str(review / "too_close_seed2_manifest.jsonl"),
                        "--project-root",
                        str(self.exp25),
                        "--output-root",
                        str(out_seed2),
                        "--limit",
                        str(too_close),
                        "--num-frames",
                        "24",
                        "--width",
                        "512",
                        "--height",
                        "288",
                        "--pcm-mode",
                        "none",
                        "--prior-mode",
                        "propainter",
                        "--no-pcm-steps",
                        "6",
                        "--no-pcm-guidance",
                        "0.0",
                        "--mask-dilation-iter",
                        "0",
                        "--seed",
                        "20260624",
                    ],
                    self.exp25,
                    gpu=gpu,
                )
            self.mark("lane_a_exp25", "completed", yield_summary=str(review / "gate32_yield_summary.json"))
        except Exception as exc:  # noqa: BLE001
            self.mark("lane_a_exp25", "failed", error=repr(exc))

    def build_gate16_manifest_from_gate32(self) -> Path:
        source = Path("/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/materialized/gate32_canonical_d0_24f/gate32_materialized.jsonl")
        out = self.run_root / "exp26_gate16_source_from_gate32.jsonl"
        if out.exists():
            return out
        rows = []
        for row in read_jsonl(source, 16):
            rows.append(
                {
                    "sample_id": f"vp2_gate16_{row['sample_id']}",
                    "source_sample_id": row["sample_id"],
                    "video_id": row["sample_id"],
                    "scene_group": row.get("scene_group", row["sample_id"]),
                    "source_dataset": "VOR-Train-BG-Gate32",
                    "source_video_path": row["winner_mp4"],
                    "winner_role": "BG",
                    "num_frames": 49,
                    "formal_49f": True,
                    "plumbing_only_13f": False,
                    "condition_definition": "winner * (1 - generated_moving_br_mask)",
                    "first_frame_gt": True,
                }
            )
        write_jsonl(out, rows)
        return out

    def lane_b_exp26(self, gpu: int) -> None:
        try:
            probe_manifest = Path("/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/49f_probe_7f9ec40/vp2_probe4_49f_masks.jsonl")
            probe_output = Path("/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623/exp26_probe4_official_inference")
            probe_review = self.run_root / "exp26_probe4_review"
            rc = self.run_logged(
                "exp26_probe4_review",
                [
                    sys.executable,
                    str(self.exp26 / "exp26_videopainter_dpo_v2" / "code" / "review_vp2_gate_outputs.py"),
                    "--manifest",
                    str(probe_manifest),
                    "--output-dir",
                    str(probe_output),
                    "--review-dir",
                    str(probe_review),
                    "--limit",
                    "4",
                ],
                self.exp26,
                gpu=None,
            )
            if rc != 0:
                self.mark("lane_b_exp26", "failed", reason="probe4_review_failed")
                return
            source_manifest = self.build_gate16_manifest_from_gate32()
            mat_root = Path("/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/gate16_from_gate32_49f")
            mat_manifest = mat_root / "vp2_gate16_49f_materialized.jsonl"
            mask_root = Path("/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/gate16_from_gate32_49f_masks")
            mask_manifest = mask_root / "vp2_gate16_49f_masks.jsonl"
            self.run_logged(
                "exp26_gate16_materialize",
                [
                    sys.executable,
                    str(self.exp26 / "exp26_videopainter_dpo_v2" / "code" / "materialize_vp2_49f_sources.py"),
                    "--manifest",
                    str(source_manifest),
                    "--source-root",
                    "/",
                    "--output-root",
                    str(mat_root / "materialized"),
                    "--output-manifest",
                    str(mat_manifest),
                    "--status-csv",
                    str(mat_root / "materialized_status.csv"),
                    "--limit",
                    "16",
                ],
                self.exp26,
                gpu=None,
            )
            self.run_logged(
                "exp26_gate16_masks",
                [
                    sys.executable,
                    str(self.exp26 / "exp26_videopainter_dpo_v2" / "code" / "generate_vp2_moving_br_masks.py"),
                    "--materialized-manifest",
                    str(mat_manifest),
                    "--output-root",
                    str(mask_root / "masks"),
                    "--output-manifest",
                    str(mask_manifest),
                    "--status-csv",
                    str(mask_root / "mask_status.csv"),
                    "--seed",
                    "20260624",
                    "--first-frame-gt",
                    "--limit",
                    "16",
                ],
                self.exp26,
                gpu=None,
            )
            vp_root = Path("/mnt/nas/hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter")
            gate16_out = self.run_root / "exp26_gate16_official_inference"
            rc = self.run_logged(
                "exp26_gate16_official_inference",
                [
                    sys.executable,
                    str(self.exp26 / "exp26_videopainter_dpo_v2" / "code" / "run_vp2_probe4_official_inference.py"),
                    "--videopainter-root",
                    str(vp_root),
                    "--base-model",
                    str(vp_root / "ckpt" / "CogVideoX-5b-I2V"),
                    "--branch-checkpoint",
                    str(vp_root / "ckpt" / "VideoPainter" / "checkpoints" / "branch"),
                    "--manifest",
                    str(mask_manifest),
                    "--output-dir",
                    str(gate16_out),
                    "--limit",
                    "16",
                    "--height",
                    "480",
                    "--width",
                    "720",
                    "--num-frames",
                    "49",
                    "--num-inference-steps",
                    "20",
                    "--dtype",
                    "bf16",
                ],
                self.exp26,
                gpu=gpu,
            )
            if rc != 0:
                return
            self.run_logged(
                "exp26_gate16_review",
                [
                    sys.executable,
                    str(self.exp26 / "exp26_videopainter_dpo_v2" / "code" / "review_vp2_gate_outputs.py"),
                    "--manifest",
                    str(mask_manifest),
                    "--output-dir",
                    str(gate16_out),
                    "--review-dir",
                    str(self.run_root / "exp26_gate16_review"),
                    "--limit",
                    "16",
                ],
                self.exp26,
                gpu=None,
            )
            self.mark("lane_b_exp26", "completed", gate16_output=str(gate16_out))
        except Exception as exc:  # noqa: BLE001
            self.mark("lane_b_exp26", "failed", error=repr(exc))

    def lane_c_exp27(self, gpu: int) -> None:
        try:
            self.run_logged(
                "exp27_sdpo_nontrivial_real_batch_parity",
                [
                    sys.executable,
                    str(self.exp27 / "exp27_paper_grounded_preference_study" / "scripts" / "run_exp27_real_batch_parity.py"),
                    "--output-dir",
                    str(self.run_root / "exp27_sdpo_nontrivial_real_batch_parity"),
                    "--mode",
                    "sdpo",
                    "--dtype",
                    "bf16",
                ],
                self.exp27,
                gpu=gpu,
            )
            self.run_logged(
                "exp27_linear_multistep_real_batch_parity",
                [
                    sys.executable,
                    str(self.exp27 / "exp27_paper_grounded_preference_study" / "scripts" / "run_exp27_real_batch_parity.py"),
                    "--output-dir",
                    str(self.run_root / "exp27_linear_multistep_real_batch_parity"),
                    "--mode",
                    "linear",
                    "--dtype",
                    "bf16",
                ],
                self.exp27,
                gpu=gpu,
            )
            self.run_logged(
                "exp27_localdpo_six_video_smoke",
                [
                    sys.executable,
                    str(self.exp27 / "exp27_paper_grounded_preference_study" / "scripts" / "run_localdpo_six_video_smoke.py"),
                    "--output-dir",
                    str(self.run_root / "exp27_localdpo_six_video_smoke"),
                    "--device",
                    "cuda",
                ],
                self.exp27,
                gpu=gpu,
            )
            self.mark("lane_c_exp27", "completed")
        except Exception as exc:  # noqa: BLE001
            self.mark("lane_c_exp27", "failed", error=repr(exc))

    def monitor_loop(self, started_at: float) -> None:
        while not self.stop:
            gpu_cp = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            app_cp = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            heartbeat = {
                "updated_at": utc_now(),
                "runtime_seconds": time.time() - started_at,
                "selected_gpus": self.args.gpus,
                "gpu_query": gpu_cp.stdout,
                "compute_apps": app_cp.stdout,
                "running": self.running,
                "tasks": read_json(self.state_path, {}).get("tasks", {}),
            }
            write_json(self.run_root / "heartbeat.json", heartbeat)
            for line in gpu_cp.stdout.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    append_csv(
                        self.run_root / "gpu_monitor.csv",
                        {"time": heartbeat["updated_at"], "gpu": parts[0], "memory_used_mib": parts[1], "memory_total_mib": parts[2], "utilization_gpu": parts[3]},
                        ["time", "gpu", "memory_used_mib", "memory_total_mib", "utilization_gpu"],
                    )
            time.sleep(self.args.interval_seconds)

    def run(self) -> int:
        write_json(
            self.run_root / "controller_identity.json",
            {
                "started_at": utc_now(),
                "pid": os.getpid(),
                "gpus": self.args.gpus,
                "exp25": str(self.exp25),
                "exp26": str(self.exp26),
                "exp27": str(self.exp27),
                "scope": "Exp25 Gate32 yield, Exp26 Probe4/Gate16, Exp27 parity/smoke; no long training.",
            },
        )
        started = time.time()
        monitor = threading.Thread(target=self.monitor_loop, args=(started,), daemon=True)
        monitor.start()
        lanes = [
            threading.Thread(target=self.lane_a_exp25, args=(self.args.gpus[0],), daemon=False),
            threading.Thread(target=self.lane_b_exp26, args=(self.args.gpus[1],), daemon=False),
            threading.Thread(target=self.lane_c_exp27, args=(self.args.gpus[2],), daemon=False),
        ]
        for lane in lanes:
            lane.start()
        for lane in lanes:
            lane.join()
        self.stop = True
        monitor.join(timeout=5)
        state = read_json(self.state_path, {})
        failed = [name for name, item in state.get("tasks", {}).items() if item.get("status") == "failed"]
        write_json(self.run_root / "final_summary.json", {"finished_at": utc_now(), "failed_tasks": failed, "state": state})
        return 0 if not failed else 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, default=Path("/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260624_three_lane"))
    p.add_argument("--snapshot-root", type=Path, default=Path("/mnt/workspace/hj/nas_hj/runtime_code_snapshots"))
    p.add_argument("--gpus", type=int, nargs=3, default=[4, 5, 6])
    p.add_argument("--interval-seconds", type=int, default=300)
    return p.parse_args()


def main() -> int:
    return Controller(parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
