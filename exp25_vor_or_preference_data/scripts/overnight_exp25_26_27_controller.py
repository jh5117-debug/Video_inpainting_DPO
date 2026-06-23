#!/usr/bin/env python3
"""Overnight controller for Exp25/Exp26/Exp27.

This is intentionally a scheduler/heartbeat layer only. It does not change the
scientific implementation of any experiment; it launches already-audited entry
points from immutable git-archive snapshots when a permitted GPU becomes idle.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GPU_ALLOWLIST = (2, 4, 5, 6)
GPU_MEM_IDLE_MIB = 5000
GPU_UTIL_IDLE = 10
CHECK_INTERVAL_SECONDS = 300
MAX_RUNTIME_SECONDS = 12 * 60 * 60
SMOKE6_SAMPLE_IDS = [
    "REAL_ENV114_00004_004_02",
    "BLENDER_FOREST039_00117",
    "REAL_ENV024_00002_008_01",
    "BLENDER_CON001_00332",
    "REAL_ENV159_00010_003_05",
    "BLENDER_FOREST039_00530",
]


@dataclass
class RunningTask:
    name: str
    pid: int
    gpu: int | None
    log_path: str
    started_at: str
    command: list[str] = field(default_factory=list)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_capture(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def latest_snapshot(root: Path, prefix: str) -> Path:
    matches = sorted([p for p in root.glob(f"{prefix}_*") if p.is_dir()])
    if not matches:
        raise FileNotFoundError(f"No snapshot found for prefix {prefix!r} under {root}")
    return matches[-1]


def query_gpus() -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    gpu_cmd = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    app_cmd = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    gpu_cp = run_capture(gpu_cmd)
    app_cp = run_capture(app_cmd)
    gpus: list[dict[str, Any]] = []
    for line in gpu_cp.stdout.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "memory_used_mib": int(float(parts[2])),
                "memory_total_mib": int(float(parts[3])),
                "utilization_gpu": int(float(parts[4])),
            }
        )
    apps: list[dict[str, Any]] = []
    for line in app_cp.stdout.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        apps.append(
            {
                "gpu_uuid": parts[0],
                "pid": parts[1],
                "process_name": parts[2],
                "used_memory_mib": parts[3],
            }
        )
    return gpus, apps, gpu_cp.stdout + "\n" + app_cp.stdout


def idle_gpus(gpus: list[dict[str, Any]], apps: list[dict[str, Any]]) -> list[int]:
    app_uuids = {a["gpu_uuid"] for a in apps}
    idle: list[int] = []
    for gpu in gpus:
        idx = gpu["index"]
        if idx not in GPU_ALLOWLIST:
            continue
        if gpu["uuid"] in app_uuids:
            continue
        if gpu["memory_used_mib"] > GPU_MEM_IDLE_MIB:
            continue
        if gpu["utilization_gpu"] > GPU_UTIL_IDLE:
            continue
        idle.append(idx)
    return idle


def load_task_state(run_root: Path) -> dict[str, Any]:
    return read_json(run_root / "task_state.json", {"tasks": {}, "running": {}})


def save_task_state(run_root: Path, state: dict[str, Any]) -> None:
    write_json(run_root / "task_state.json", state)


def mark_task(run_root: Path, name: str, status: str, **extra: Any) -> None:
    state = load_task_state(run_root)
    tasks = state.setdefault("tasks", {})
    item = tasks.setdefault(name, {})
    item.update({"status": status, "updated_at": utc_now(), **extra})
    save_task_state(run_root, state)


def task_status(run_root: Path, name: str) -> str:
    state = load_task_state(run_root)
    return state.get("tasks", {}).get(name, {}).get("status", "queued")


def prepare_exp25_smoke6(exp25: Path, run_root: Path) -> dict[str, str]:
    fixed_manifest = run_root / "exp25_fixed_smoke6_gate128_member_manifest.jsonl"
    materialized_root = Path("/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/materialized/gate128_smoke6_canonical_d0_24f")
    materialized_manifest = materialized_root / "smoke6_materialized.jsonl"
    source_manifest = exp25 / "exp25_vor_or_preference_data" / "manifests" / "vor_gate128.jsonl"
    if not fixed_manifest.exists():
        wanted = {sid: None for sid in SMOKE6_SAMPLE_IDS}
        rows = []
        with source_manifest.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                sid = row.get("sample_id")
                if sid in wanted:
                    wanted[sid] = row
        missing = [sid for sid, row in wanted.items() if row is None]
        if missing:
            raise RuntimeError(f"fixed Smoke6 sample IDs not found in Gate128: {missing}")
        with fixed_manifest.open("w", encoding="utf-8") as f:
            for sid in SMOKE6_SAMPLE_IDS:
                f.write(json.dumps(wanted[sid], sort_keys=True) + "\n")
    ok_rows = 0
    if materialized_manifest.exists():
        ok_rows = sum(1 for line in materialized_manifest.open("r", encoding="utf-8") if line.strip())
    if ok_rows < len(SMOKE6_SAMPLE_IDS):
        cmd = [
            sys.executable,
            str(exp25 / "exp25_vor_or_preference_data" / "scripts" / "materialize_vor_or_inputs.py"),
            "--manifest",
            str(fixed_manifest),
            "--extraction-root",
            "/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/extracted/vor_gate128_exact_20260623",
            "--output-root",
            str(materialized_root),
            "--output-manifest",
            str(materialized_manifest),
            "--limit",
            str(len(SMOKE6_SAMPLE_IDS)),
            "--frames",
            "24",
        ]
        cp = run_capture(cmd, cwd=exp25, timeout=1800)
        (run_root / "exp25_materialize_smoke6.log").write_text("[cmd] " + " ".join(cmd) + "\n" + cp.stdout, encoding="utf-8")
        if cp.returncode != 0:
            raise RuntimeError(f"Exp25 Smoke6 materialization failed, rc={cp.returncode}")
    return {"manifest": str(materialized_manifest), "materialized_root": str(materialized_root)}


def run_cpu_exp25_prepare(run_root: Path, snapshot_root: Path) -> None:
    if task_status(run_root, "exp25_prepare_smoke6") == "completed":
        return
    exp25 = latest_snapshot(snapshot_root, "exp25")
    try:
        payload = prepare_exp25_smoke6(exp25, run_root)
        mark_task(run_root, "exp25_prepare_smoke6", "completed", **payload)
    except Exception as exc:  # noqa: BLE001
        mark_task(run_root, "exp25_prepare_smoke6", "blocked", error=repr(exc))


def run_cpu_effecterase_inventory(run_root: Path, snapshot_root: Path) -> None:
    if task_status(run_root, "exp25_effecterase_inventory") in {"completed", "blocked"}:
        return
    exp25 = latest_snapshot(snapshot_root, "exp25")
    cmd = [
        sys.executable,
        str(exp25 / "exp25_vor_or_preference_data" / "scripts" / "audit_hf_effecterase_repo.py"),
        "--probe-readme",
    ]
    cp = run_capture(cmd, cwd=exp25, timeout=1800)
    log = run_root / "exp25_effecterase_inventory.log"
    log.write_text("[cmd] " + " ".join(cmd) + "\n" + cp.stdout, encoding="utf-8")
    if cp.returncode == 0:
        mark_task(run_root, "exp25_effecterase_inventory", "completed", log_path=str(log))
    else:
        mark_task(run_root, "exp25_effecterase_inventory", "blocked", log_path=str(log), returncode=cp.returncode)


def run_cpu_exp27_parity(run_root: Path, snapshot_root: Path) -> None:
    if task_status(run_root, "exp27_cpu_parity_refresh") == "completed":
        return
    exp27 = latest_snapshot(snapshot_root, "exp27")
    out_dir = run_root / "exp27_cpu_parity_refresh"
    cmd = [
        sys.executable,
        str(exp27 / "exp27_paper_grounded_preference_study" / "scripts" / "run_exp27_cpu_parity.py"),
        "--output-dir",
        str(out_dir),
    ]
    cp = run_capture(cmd, cwd=exp27, timeout=1800)
    log = run_root / "exp27_cpu_parity_refresh.log"
    log.write_text("[cmd] " + " ".join(cmd) + "\n" + cp.stdout, encoding="utf-8")
    if cp.returncode == 0:
        mark_task(run_root, "exp27_cpu_parity_refresh", "completed", log_path=str(log), output_dir=str(out_dir))
    else:
        mark_task(run_root, "exp27_cpu_parity_refresh", "blocked", log_path=str(log), returncode=cp.returncode)


def launch_gpu_task(run_root: Path, task_name: str, gpu: int, cmd: list[str], cwd: Path, env_extra: dict[str, str] | None = None) -> RunningTask:
    log_path = run_root / f"{task_name}.log"
    env = os.environ.copy()
    env.update(env_extra or {})
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[start {utc_now()}] gpu={gpu}\n[cmd] {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    running = RunningTask(name=task_name, pid=proc.pid, gpu=gpu, log_path=str(log_path), started_at=utc_now(), command=cmd)
    state = load_task_state(run_root)
    state.setdefault("running", {})[task_name] = running.__dict__
    state.setdefault("tasks", {}).setdefault(task_name, {}).update({"status": "running", "updated_at": utc_now(), "pid": proc.pid, "gpu": gpu, "log_path": str(log_path)})
    save_task_state(run_root, state)
    return running


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def refresh_running(run_root: Path) -> None:
    state = load_task_state(run_root)
    running = dict(state.get("running", {}))
    changed = False
    for name, info in list(running.items()):
        pid = int(info.get("pid", -1))
        if pid > 0 and process_alive(pid):
            continue
        log_path = Path(info.get("log_path", ""))
        status = "completed"
        if log_path.exists():
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
            if any(token in tail for token in ["Traceback", "FAILED", "RuntimeError", "ERROR", "OutOfMemory", "CUDA out of memory"]):
                status = "failed"
        state.setdefault("tasks", {}).setdefault(name, {}).update({"status": status, "updated_at": utc_now(), "finished_pid": pid})
        running.pop(name, None)
        changed = True
    if changed:
        state["running"] = running
        save_task_state(run_root, state)


def maybe_launch_exp25_smoke6(run_root: Path, snapshot_root: Path, idle: list[int]) -> None:
    if not idle:
        return
    if task_status(run_root, "exp25_prepare_smoke6") != "completed":
        return
    if task_status(run_root, "exp25_smoke6_d0") not in {"queued", "failed"}:
        return
    exp25 = latest_snapshot(snapshot_root, "exp25")
    prep = load_task_state(run_root).get("tasks", {}).get("exp25_prepare_smoke6", {})
    output_root = Path("/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/preference_candidates/smoke6_canonical_raw6_d0")
    cmd = [
        sys.executable,
        str(exp25 / "exp25_vor_or_preference_data" / "scripts" / "run_vor_or_model_smoke.py"),
        "--model",
        "diffueraser",
        "--manifest",
        str(prep["manifest"]),
        "--project-root",
        str(exp25),
        "--output-root",
        str(output_root),
        "--limit",
        "6",
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
    ]
    launch_gpu_task(run_root, "exp25_smoke6_d0", idle[0], cmd, exp25)


def write_heartbeat(run_root: Path, gpus: list[dict[str, Any]], apps: list[dict[str, Any]], idle: list[int], started_at: float) -> None:
    state = load_task_state(run_root)
    payload = {
        "updated_at": utc_now(),
        "runtime_seconds": time.time() - started_at,
        "gpu_allowlist": GPU_ALLOWLIST,
        "idle_gpus": idle,
        "gpus": gpus,
        "apps": apps,
        "tasks": state.get("tasks", {}),
        "running": state.get("running", {}),
    }
    write_json(run_root / "heartbeat.json", payload)
    for gpu in gpus:
        append_csv(
            run_root / "gpu_timeline.csv",
            {
                "time": payload["updated_at"],
                "index": gpu["index"],
                "memory_used_mib": gpu["memory_used_mib"],
                "memory_total_mib": gpu["memory_total_mib"],
                "utilization_gpu": gpu["utilization_gpu"],
                "idle": gpu["index"] in idle,
            },
            ["time", "index", "memory_used_mib", "memory_total_mib", "utilization_gpu", "idle"],
        )


def write_queue(run_root: Path) -> None:
    state = load_task_state(run_root)
    rows = []
    priority = [
        "exp25_prepare_smoke6",
        "exp25_effecterase_inventory",
        "exp27_cpu_parity_refresh",
        "exp25_smoke6_d0",
        "exp26_probe4_official_inference",
        "exp27_sdpo_real_batch_parity",
        "exp27_linear_real_batch_parity",
        "exp25_gate32",
    ]
    for name in priority:
        item = state.get("tasks", {}).get(name, {})
        rows.append({"task": name, "status": item.get("status", "queued"), "pid": item.get("pid", ""), "gpu": item.get("gpu", ""), "updated_at": item.get("updated_at", "")})
    path = run_root / "overnight_queue.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "status", "pid", "gpu", "updated_at"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623"))
    parser.add_argument("--snapshot-root", type=Path, default=Path("/mnt/workspace/hj/nas_hj/runtime_code_snapshots"))
    parser.add_argument("--max-seconds", type=int, default=MAX_RUNTIME_SECONDS)
    parser.add_argument("--interval-seconds", type=int, default=CHECK_INTERVAL_SECONDS)
    args = parser.parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    (args.run_root / "controller_pid.txt").write_text(str(os.getpid()) + "\n", encoding="utf-8")
    started_at = time.time()
    mark_task(args.run_root, "controller", "running", pid=os.getpid(), max_seconds=args.max_seconds)

    stop = False

    def handle_stop(signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True
        mark_task(args.run_root, "controller", "stopping", signal=signum)

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)

    while not stop and time.time() - started_at < args.max_seconds:
        refresh_running(args.run_root)
        run_cpu_exp25_prepare(args.run_root, args.snapshot_root)
        run_cpu_effecterase_inventory(args.run_root, args.snapshot_root)
        run_cpu_exp27_parity(args.run_root, args.snapshot_root)
        gpus, apps, _raw = query_gpus()
        idle = idle_gpus(gpus, apps)
        maybe_launch_exp25_smoke6(args.run_root, args.snapshot_root, idle)
        write_heartbeat(args.run_root, gpus, apps, idle, started_at)
        write_queue(args.run_root)
        time.sleep(args.interval_seconds)

    mark_task(args.run_root, "controller", "completed" if not stop else "stopped")
    write_queue(args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
