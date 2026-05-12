#!/usr/bin/env python3
"""Best-effort W&B lifecycle marker for Slurm-launched jobs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re


def _tail(path: str | None, lines: int) -> str:
    if not path:
        return ""
    file_path = Path(path)
    if not file_path.is_file():
        return ""
    with file_path.open("r", encoding="utf-8", errors="replace") as handle:
        data = handle.readlines()
    return "".join(data[-lines:])


def _extract_reason(log_tail: str, event: str) -> str:
    """Pull a compact failure hint from the copied Slurm/training log tail."""
    if not log_tail:
        return f"{event}: no log tail available"

    patterns = [
        r"CUDA out of memory[^\n]*",
        r"MisconfigurationException: [^\n]*",
        r"Failed to get device handle for GPU [^\n]*",
        r"\[vc2-train\]\[gpu-preflight\]\[ERROR\][^\n]*",
        r"torch\.cuda\.is_available\(\)=False[^\n]*",
        r"RuntimeError: [^\n]*",
        r"ValueError: [^\n]*",
        r"KeyError: [^\n]*",
        r"FileNotFoundError: [^\n]*",
        r"ModuleNotFoundError: [^\n]*",
        r"ImportError: [^\n]*",
        r"AssertionError[^\n]*",
        r"slurmstepd:[^\n]*",
        r"\[vc2-train\]\[error\][^\n]*",
        r"\[vc2-train\]\[preflight\]\[ERROR\][^\n]*",
        r"Traceback \(most recent call last\):",
        r"FAILED[^\n]*",
        r"CANCELLED[^\n]*",
        r"TIMEOUT[^\n]*",
    ]
    lines = [line.strip() for line in log_tail.splitlines() if line.strip()]
    for pattern in patterns:
        for clean in reversed(lines):
            match = re.search(pattern, clean)
            if match:
                return match.group(0)[:500]
    return lines[-1][:500] if lines else f"{event}: no non-empty log lines"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", required=True, choices=["start", "running", "failed", "finished"])
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--group", default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--tail-lines", type=int, default=200)
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--config", action="append", default=[])
    args = parser.parse_args()

    try:
        import wandb
    except Exception as exc:  # pragma: no cover - this script is best effort.
        print(f"[wandb-event][warn] cannot import wandb: {type(exc).__name__}: {exc}")
        return 0

    config = {}
    for item in args.config:
        if "=" in item:
            key, value = item.split("=", 1)
            config[key] = value

    try:
        run = wandb.init(
            project=args.project,
            entity=args.entity or None,
            group=args.group or None,
            name=args.run_name,
            id=args.run_id,
            resume="allow",
            dir=args.run_dir,
            job_type="vc2_train",
            config=config or None,
        )
        run.summary["launcher_event"] = args.event
        run.summary["launcher_last_event"] = args.event
        run.summary["launcher_exit_code"] = args.exit_code
        run.summary["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", "")
        run.summary["slurm_job_partition"] = os.environ.get("SLURM_JOB_PARTITION", "")

        log_tail = _tail(args.log_file, args.tail_lines)
        if log_tail:
            print(f"[wandb-event] launcher log tail ({args.event}, last {args.tail_lines} lines) BEGIN", flush=True)
            print(log_tail[-20000:], flush=True)
            print(f"[wandb-event] launcher log tail ({args.event}) END", flush=True)
            tail_dir = Path(args.run_dir) / "wandb_tail"
            tail_dir.mkdir(parents=True, exist_ok=True)
            tail_path = tail_dir / f"{args.run_id}.{args.event}.tail.txt"
            tail_path.write_text(log_tail, encoding="utf-8")
            wandb.save(str(tail_path), policy="now")
            run.summary["launcher_log_tail_file"] = str(tail_path)
            run.summary["launcher_log_tail_preview"] = log_tail[-3500:]
            if args.event == "failed":
                run.summary["launcher_error_reason"] = _extract_reason(log_tail, args.event)
            try:
                tail_preview = log_tail[-10000:]
                tail_table = wandb.Table(
                    columns=["event", "exit_code", "tail"],
                    data=[[args.event, args.exit_code, tail_preview]],
                )
                wandb.log({"launcher/log_tail": tail_table})
            except Exception as exc:
                print(f"[wandb-event][warn] failed to log tail table: {type(exc).__name__}: {exc}")

        wandb.log({"launcher/event_code": {"start": 0, "running": 0.5, "failed": 1, "finished": 2}[args.event]})
        wandb.finish(exit_code=args.exit_code)
        print(f"[wandb-event] logged event={args.event} run_id={args.run_id}")
    except Exception as exc:  # pragma: no cover - keep Slurm failure reason primary.
        print(f"[wandb-event][warn] failed to log event: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
