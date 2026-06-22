#!/usr/bin/env python3
"""Monitor Exp25 EffectErase HAL->PAI transfer and write lightweight reports."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

try:
    from effecterase_common import (
        DEFAULT_PAI_HOST,
        DEFAULT_PAI_KEY,
        LOGS,
        REPORTS,
        RUNTIME,
        ensure_dirs,
        free_bytes,
        read_json,
        ssh_cmd,
    )
except ModuleNotFoundError:
    from .effecterase_common import (
        DEFAULT_PAI_HOST,
        DEFAULT_PAI_KEY,
        LOGS,
        REPORTS,
        RUNTIME,
        ensure_dirs,
        free_bytes,
        read_json,
        ssh_cmd,
    )


def count_manifest(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    files = 0
    bytes_ = 0
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "VERIFIED":
                files += 1
                bytes_ += int(row.get("size") or 0)
    return files, bytes_


def pid_alive(pid: int) -> bool:
    try:
        Path(f"/proc/{pid}").stat()
        return True
    except FileNotFoundError:
        return False


def exp23_readonly_status(key: Path, host: str) -> dict:
    cmd = (
        "PAIR=phaseA_scale1_pair001_outer2_corrected_outer_control_seed20260619_gpus2456; "
        "BASE=/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp23_two_stage_pool_morphology_sweep/pairs/$PAIR; "
        "echo PROCS=$(ps -ef | grep -E 'Phy|train_exp23_stage|run_davis50' | grep -v grep | wc -l); "
        "for f in $BASE/*/*/region_diagnostics.csv; do [ -f \"$f\" ] && echo REGION:$f:$(tail -1 \"$f\" | cut -d, -f1-3); done; "
        "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits -i 2,4,5,6 2>/dev/null || true"
    )
    cp = subprocess.run(ssh_cmd(key, host, cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
    return {"returncode": cp.returncode, "stdout": cp.stdout[-4000:], "stderr": cp.stderr[-1000:]}


def render_status(state: dict, hb: dict, completed_files: int, completed_bytes: int, exp23: dict) -> str:
    total_files = int(state.get("total_files") or 0)
    total_bytes = int(state.get("total_bytes") or 0)
    pct = 100.0 * completed_bytes / total_bytes if total_bytes else 0.0
    return "\n".join([
        "# Exp25 EffectErase Transfer Status",
        "",
        f"- timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        f"- controller_pid: {state.get('pid')}",
        f"- controller_alive: {pid_alive(int(state.get('pid') or 0)) if state.get('pid') else False}",
        f"- status: {state.get('status')}",
        f"- phase: {hb.get('current_phase') or state.get('current_phase')}",
        f"- current_file: `{hb.get('current_filename') or state.get('current_filename') or ''}`",
        f"- completed: {completed_files}/{total_files}",
        f"- completed_bytes: {completed_bytes}/{total_bytes}",
        f"- percent: {pct:.4f}",
        f"- last_error: `{hb.get('last_error') or state.get('last_error') or ''}`",
        f"- hal_staging: `{state.get('hal_staging')}`",
        f"- pai_download_dir: `{state.get('pai_download_dir')}`",
        "",
        "## Exp23 Read-Only Status",
        "",
        "```",
        exp23.get("stdout", "").strip(),
        exp23.get("stderr", "").strip(),
        "```",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--pai-host", default=DEFAULT_PAI_HOST)
    parser.add_argument("--pai-key", type=Path, default=DEFAULT_PAI_KEY)
    args = parser.parse_args()
    ensure_dirs()
    log = LOGS / "effecterase_transfer_monitor.log"
    while True:
        state = read_json(RUNTIME / "transfer_state.json", {})
        hb = read_json(RUNTIME / "transfer_heartbeat.json", {})
        completed_files, completed_bytes = count_manifest(RUNTIME / "transfer_manifest.csv")
        if state.get("hal_staging"):
            try:
                state["hal_free_bytes"] = free_bytes(Path(state["hal_staging"]))
            except Exception:
                pass
        exp23 = exp23_readonly_status(args.pai_key, args.pai_host)
        snapshot = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "state": state,
            "heartbeat": hb,
            "completed_files": completed_files,
            "completed_bytes": completed_bytes,
            "exp23": exp23,
        }
        (RUNTIME / "monitor_state.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
        (REPORTS / "effecterase_download_status.md").write_text(render_status(state, hb, completed_files, completed_bytes, exp23))
        with log.open("a") as f:
            f.write(json.dumps(snapshot, sort_keys=True) + "\n")
        if args.once:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
