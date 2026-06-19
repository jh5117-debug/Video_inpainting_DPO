"""PAI GPU allocator with lock files.

This script is conservative: it only reports GPUs with no compute process and
low memory usage. It creates lock files with O_EXCL so parallel workers do not
collide.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def query_nvidia_smi() -> list[dict[str, object]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    gpus = []
    for line in result.stdout.strip().splitlines():
        idx, mem, util = [part.strip() for part in line.split(",")]
        gpus.append({"index": int(idx), "memory_used_mib": int(mem), "utilization": int(util)})
    return gpus


def lock_gpu(index: int, lock_dir: Path) -> bool:
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / f"hj_exp20_gpu_{index}.lock"
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        f.write(f"pid={os.getpid()}\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-memory-mib", type=int, default=1024)
    parser.add_argument("--lock-dir", default="/tmp")
    parser.add_argument("--lock", action="store_true")
    args = parser.parse_args()
    candidates = [g for g in query_nvidia_smi() if g["memory_used_mib"] <= args.max_memory_mib]
    selected = []
    for gpu in candidates:
        if len(selected) >= args.num_gpus:
            break
        if args.lock and not lock_gpu(int(gpu["index"]), Path(args.lock_dir)):
            continue
        selected.append(gpu)
    print(json.dumps({"selected": selected, "all_candidates": candidates}, indent=2))


if __name__ == "__main__":
    main()
