"""Conservative PAI GPU allocator for Exp20 trials."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def query_gpus() -> list[dict[str, Any]]:
    out = _run([
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    rows = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        idx, mem, util = [x.strip() for x in line.split(",")]
        rows.append({"index": int(idx), "memory_used_mib": int(mem), "utilization_gpu": int(util)})
    return rows


def query_compute_pids() -> dict[int, list[str]]:
    try:
        out = _run([
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ])
    except subprocess.CalledProcessError:
        return {}
    # Map by index via plain nvidia-smi process table is unreliable across driver
    # versions; use memory/process absence as conservative filter and preserve raw.
    raw = [line.strip() for line in out.splitlines() if line.strip()]
    return {-1: raw} if raw else {}


def sample_idle(
    samples: int,
    interval_seconds: float,
    max_memory_mib: int,
    excluded: set[int],
) -> tuple[set[int], list[dict[str, Any]]]:
    stable: set[int] | None = None
    history: list[dict[str, Any]] = []
    for _ in range(samples):
        gpus = query_gpus()
        compute = query_compute_pids()
        compute_present = bool(compute)
        idle_now = {
            int(g["index"])
            for g in gpus
            if int(g["index"]) not in excluded
            and int(g["memory_used_mib"]) <= max_memory_mib
            and int(g["utilization_gpu"]) <= 5
            and not compute_present
        }
        history.append({"gpus": gpus, "compute_processes_raw": compute, "idle_now": sorted(idle_now)})
        stable = idle_now if stable is None else stable & idle_now
        if interval_seconds > 0:
            time.sleep(interval_seconds)
    return stable or set(), history


def acquire_locks(gpu_ids: list[int], lock_dir: Path, owner: str) -> list[Path]:
    lock_dir.mkdir(parents=True, exist_ok=True)
    acquired: list[Path] = []
    try:
        for gpu_id in gpu_ids:
            path = lock_dir / f"exp20_gpu_{gpu_id}.lock"
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as handle:
                handle.write(json.dumps({"pid": os.getpid(), "gpu": gpu_id, "owner": owner}) + "\n")
            acquired.append(path)
    except Exception:
        for path in acquired:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    return acquired


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-memory-mib", type=int, default=1024)
    parser.add_argument("--exclude", default="7")
    parser.add_argument("--lock-dir", default="/tmp/exp20_gpu_locks")
    parser.add_argument("--owner", default="exp20")
    parser.add_argument("--lock", action="store_true")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    args = parser.parse_args()

    excluded = {int(x) for x in args.exclude.split(",") if x.strip()}
    stable, history = sample_idle(args.samples, args.interval_seconds, args.max_memory_mib, excluded)
    selected = sorted(stable)[: args.num_gpus]
    payload: dict[str, Any] = {
        "selected": selected,
        "requested": args.num_gpus,
        "excluded": sorted(excluded),
        "max_memory_mib": args.max_memory_mib,
        "history": history,
        "locks": [],
    }
    if len(selected) < args.num_gpus:
        payload["status"] = "INSUFFICIENT_IDLE_GPU"
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    if args.lock:
        locks = acquire_locks(selected, Path(args.lock_dir), args.owner)
        payload["locks"] = [str(p) for p in locks]
    payload["status"] = "GPU_LOCKED" if args.lock else "GPU_AVAILABLE"
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
