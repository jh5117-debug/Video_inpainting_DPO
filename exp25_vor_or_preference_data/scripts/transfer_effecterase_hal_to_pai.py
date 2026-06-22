#!/usr/bin/env python3
"""Resumable one-file-at-a-time HAL -> PAI EffectErase VOR transfer."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

try:
    from effecterase_common import (
        DEFAULT_HF_AUTH_HOME,
        DEFAULT_HF_ENV,
        DEFAULT_PAI_DOWNLOAD_DIR,
        DEFAULT_PAI_HOST,
        DEFAULT_PAI_KEY,
        LOGS,
        REGISTRY,
        REPORTS,
        RUNTIME,
        FileState,
        append_csv,
        atomic_write_json,
        ensure_dirs,
        free_bytes,
        now_iso,
        read_json,
        safe_job_name,
        safe_remove_tree,
        sha256_file,
        shquote,
        ssh_cmd,
        total_bytes,
    )
except ModuleNotFoundError:
    from .effecterase_common import (
        DEFAULT_HF_AUTH_HOME,
        DEFAULT_HF_ENV,
        DEFAULT_PAI_DOWNLOAD_DIR,
        DEFAULT_PAI_HOST,
        DEFAULT_PAI_KEY,
        LOGS,
        REGISTRY,
        REPORTS,
        RUNTIME,
        FileState,
        append_csv,
        atomic_write_json,
        ensure_dirs,
        free_bytes,
        now_iso,
        read_json,
        safe_job_name,
        safe_remove_tree,
        sha256_file,
        shquote,
        ssh_cmd,
        total_bytes,
    )


MANIFEST_FIELDS = [
    "timestamp",
    "filename",
    "group",
    "size",
    "status",
    "hal_sha256",
    "pai_sha256",
    "pai_final_path",
    "hal_cleanup_bytes",
    "hal_free_after_cleanup",
    "attempt",
    "last_error",
]


def remote_stat(key: Path, host: str, path: str) -> tuple[bool, int]:
    cmd = f"if [ -f {shquote(path)} ]; then stat -c %s {shquote(path)}; else echo MISSING; fi"
    cp = subprocess.run(ssh_cmd(key, host, cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if cp.returncode != 0:
        raise RuntimeError(cp.stderr.strip())
    out = cp.stdout.strip()
    if out == "MISSING" or not out:
        return False, 0
    return True, int(out.splitlines()[-1])


def remote_sha256(key: Path, host: str, path: str) -> str:
    cp = subprocess.run(ssh_cmd(key, host, f"sha256sum {shquote(path)} | awk '{{print $1}}'"),
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if cp.returncode != 0:
        raise RuntimeError(cp.stderr.strip())
    return cp.stdout.strip().splitlines()[-1]


def remote_mkdir(key: Path, host: str, path: str) -> None:
    subprocess.run(ssh_cmd(key, host, f"mkdir -p {shquote(path)}"), check=True)


def remote_finalize(key: Path, host: str, partial: str, final: str) -> None:
    parent = str(Path(final).parent)
    cmd = f"mkdir -p {shquote(parent)} && mv -f {shquote(partial)} {shquote(final)} && sync"
    subprocess.run(ssh_cmd(key, host, cmd), check=True)


def remote_df_bytes(key: Path, host: str, path: str) -> int:
    cp = subprocess.run(ssh_cmd(key, host, f"mkdir -p {shquote(path)} && df -B1 {shquote(path)} | tail -1 | awk '{{print $4}}'"),
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if cp.returncode != 0:
        return 0
    try:
        return int(cp.stdout.strip().splitlines()[-1])
    except Exception:
        return 0


def load_completed(manifest: Path) -> dict[str, dict]:
    completed = {}
    if not manifest.exists():
        return completed
    import csv
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "VERIFIED":
                completed[row["filename"]] = row
    return completed


def write_heartbeat(state: dict) -> None:
    hb = dict(state)
    hb["heartbeat_at"] = now_iso()
    atomic_write_json(RUNTIME / "transfer_heartbeat.json", hb)


def download_one(args, revision: str, item: dict, job_dir: Path, state: dict) -> Path:
    download_dir = job_dir / "download"
    cache_hub = job_dir / "cache" / "hub"
    cache_xet = job_dir / "cache" / "xet"
    download_dir.mkdir(parents=True, exist_ok=True)
    cache_hub.mkdir(parents=True, exist_ok=True)
    cache_xet.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "HF_HOME": str(args.hf_auth_home),
        "HF_HUB_CACHE": str(cache_hub),
        "HF_XET_CACHE": str(cache_xet),
        "HF_HUB_DOWNLOAD_TIMEOUT": "120",
    })
    log_path = job_dir / "download.log"
    cmd = [
        str(args.hf_env / "bin" / "hf"),
        "download",
        args.repo_id,
        item["filename"],
        "--repo-type",
        "dataset",
        "--revision",
        revision,
        "--local-dir",
        str(download_dir),
    ]
    state.update({"current_phase": "HAL_DOWNLOAD", "current_filename": item["filename"], "download_command": " ".join(shlex.quote(c) for c in cmd)})
    write_heartbeat(state)
    with log_path.open("ab") as log:
        subprocess.run(["nice", "-n", "15", "ionice", "-c2", "-n7", *cmd], env=env, check=True, stdout=log, stderr=log)
    local = download_dir / item["filename"]
    if not local.exists():
        matches = list(download_dir.rglob(Path(item["filename"]).name))
        if len(matches) == 1:
            local = matches[0]
    if not local.exists():
        raise RuntimeError(f"Downloaded file not found for {item['filename']}")
    if local.is_symlink():
        target = local.resolve()
        copy_path = local.with_suffix(local.suffix + ".materialized")
        copy_path.write_bytes(target.read_bytes())
        local.unlink()
        copy_path.replace(local)
    size = local.stat().st_size
    if size != int(item["size"]):
        raise RuntimeError(f"HAL size mismatch for {item['filename']}: {size} != {item['size']}")
    return local


def rsync_one(args, local: Path, partial: str, state: dict) -> None:
    remote = f"{args.pai_host}:{partial}"
    cmd = [
        "rsync",
        "-avh",
        "--partial",
        "--append-verify",
        "--info=progress2",
        "--bwlimit=40000",
        "-e",
        f"ssh -i {args.pai_key} -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=30",
        str(local),
        remote,
    ]
    state.update({"current_phase": "RSYNC_TO_PAI", "rsync_command": " ".join(shlex.quote(c) for c in cmd)})
    write_heartbeat(state)
    with (LOGS / "effecterase_rsync.log").open("ab") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=log)


def process_file(args, revision: str, item: dict, completed: dict[str, dict], state: dict) -> bool:
    filename = item["filename"]
    final_dir = str(args.pai_download_dir / revision / str(Path(filename).parent))
    if final_dir.endswith("/."):
        final_dir = str(args.pai_download_dir / revision)
    final_path = str(args.pai_download_dir / revision / filename)
    partial_path = str(Path(final_path).parent / ("." + Path(filename).name + ".partial"))

    if filename in completed:
        exists, size = remote_stat(args.pai_key, args.pai_host, final_path)
        if exists and size == int(item["size"]) and completed[filename].get("hal_sha256") == completed[filename].get("pai_sha256"):
            state.update({"current_phase": "SKIPPED_VERIFIED", "current_filename": filename})
            write_heartbeat(state)
            return True

    job = args.hal_staging / "jobs" / safe_job_name(filename)
    job.mkdir(parents=True, exist_ok=True)
    fs = FileState(filename=filename, group=item["group"], size=int(item["size"]), started_at=now_iso())
    remote_mkdir(args.pai_key, args.pai_host, final_dir)

    max_download_retries = 5
    for attempt in range(1, max_download_retries + 1):
        fs.retries = attempt - 1
        try:
            hal_free_before = free_bytes(args.hal_staging)
            state.update({
                "current_filename": filename,
                "current_group": item["group"],
                "current_size": int(item["size"]),
                "current_attempt": attempt,
                "hal_free_before_job": hal_free_before,
                "pai_free_bytes": remote_df_bytes(args.pai_key, args.pai_host, str(args.pai_download_dir)),
            })
            write_heartbeat(state)
            local = download_one(args, revision, item, job, state)
            hal_sha = sha256_file(local)
            fs.hal_sha256 = hal_sha
            state.update({"current_phase": "HAL_SHA256_DONE", "hal_sha256": hal_sha, "hal_file_size": local.stat().st_size})
            write_heartbeat(state)

            for rsync_attempt in range(1, 999999):
                try:
                    rsync_one(args, local, partial_path, state)
                    break
                except subprocess.CalledProcessError as e:
                    state.update({"current_phase": "RSYNC_RETRY_WAIT", "last_error": str(e), "rsync_attempt": rsync_attempt})
                    write_heartbeat(state)
                    time.sleep(min(300, 60 * rsync_attempt))

            exists, pai_size = remote_stat(args.pai_key, args.pai_host, partial_path)
            if not exists or pai_size != int(item["size"]):
                raise RuntimeError(f"PAI partial size mismatch for {filename}: {pai_size} != {item['size']}")
            pai_sha = remote_sha256(args.pai_key, args.pai_host, partial_path)
            fs.pai_sha256 = pai_sha
            if hal_sha != pai_sha:
                ts = int(time.time())
                bad_path = str(Path(partial_path).parent / (".bad." + str(ts) + "." + Path(filename).name))
                subprocess.run(ssh_cmd(args.pai_key, args.pai_host, f"mv -f {shquote(partial_path)} {shquote(bad_path)}"), check=True)
                raise RuntimeError(f"BLOCKED_SHA256_MISMATCH {filename}: {hal_sha} != {pai_sha}")
            remote_finalize(args.pai_key, args.pai_host, partial_path, final_path)
            fs.status = "VERIFIED"
            fs.completed_at = now_iso()
            cleanup_bytes = safe_remove_tree(job, args.hal_staging / "jobs")
            row = {
                "timestamp": fs.completed_at,
                "filename": filename,
                "group": fs.group,
                "size": fs.size,
                "status": fs.status,
                "hal_sha256": fs.hal_sha256,
                "pai_sha256": fs.pai_sha256,
                "pai_final_path": final_path,
                "hal_cleanup_bytes": cleanup_bytes,
                "hal_free_after_cleanup": free_bytes(args.hal_staging),
                "attempt": attempt,
                "last_error": "",
            }
            append_csv(args.manifest, MANIFEST_FIELDS, row)
            append_csv(REPORTS / "effecterase_hal_to_pai_transfer.csv", MANIFEST_FIELDS, row)
            append_csv(REGISTRY / "download_manifest.csv", MANIFEST_FIELDS, row)
            state.update({"current_phase": "VERIFIED", "current_filename": filename, "last_verified": row, "last_error": ""})
            write_heartbeat(state)
            return True
        except Exception as e:
            fs.last_error = str(e)
            state.update({"current_phase": "ERROR_RETRY_WAIT", "current_filename": filename, "last_error": str(e), "current_attempt": attempt})
            write_heartbeat(state)
            if "401" in str(e) or "403" in str(e):
                raise RuntimeError(f"HF_ACCESS_LOST: {e}") from e
            if attempt >= max_download_retries or "BLOCKED_SHA256_MISMATCH" in str(e):
                raise
            time.sleep([30, 60, 120, 240, 480][attempt - 1])
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="FudanCVL/EffectErase")
    parser.add_argument("--inventory", default="reports/effecterase_remote_inventory.json")
    parser.add_argument("--hf-env", type=Path, default=DEFAULT_HF_ENV)
    parser.add_argument("--hf-auth-home", type=Path, default=DEFAULT_HF_AUTH_HOME)
    parser.add_argument("--hal-staging", type=Path, default=None)
    parser.add_argument("--pai-host", default=DEFAULT_PAI_HOST)
    parser.add_argument("--pai-key", type=Path, default=DEFAULT_PAI_KEY)
    parser.add_argument("--pai-download-dir", type=Path, default=DEFAULT_PAI_DOWNLOAD_DIR)
    parser.add_argument("--manifest", type=Path, default=RUNTIME / "transfer_manifest.csv")
    args = parser.parse_args()
    ensure_dirs()
    if args.hal_staging is None:
        selected = json.loads((RUNTIME / "selected_hal_staging.json").read_text())
        args.hal_staging = Path(selected["path"])
    args.hal_staging.mkdir(parents=True, exist_ok=True)
    (args.hal_staging / "jobs").mkdir(parents=True, exist_ok=True)

    inv = read_json(Path(args.inventory))
    if not inv:
        raise SystemExit("Missing remote inventory")
    revision = inv["revision"]
    required = inv["required_files"]
    completed = load_completed(args.manifest)
    state = {
        "pid": os.getpid(),
        "pgid": os.getpgid(0),
        "started_at": now_iso(),
        "status": "RUNNING",
        "dataset_revision": revision,
        "total_files": len(required),
        "total_bytes": total_bytes(required),
        "completed_files": len(completed),
        "completed_bytes": sum(int(v["size"]) for v in completed.values()),
        "hal_staging": str(args.hal_staging),
        "pai_download_dir": str(args.pai_download_dir / revision),
    }
    atomic_write_json(RUNTIME / "transfer_state.json", state)
    write_heartbeat(state)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    for item in required:
        completed = load_completed(args.manifest)
        state["completed_files"] = len(completed)
        state["completed_bytes"] = sum(int(v["size"]) for v in completed.values())
        atomic_write_json(RUNTIME / "transfer_state.json", state)
        process_file(args, revision, item, completed, state)
        completed = load_completed(args.manifest)
        state["completed_files"] = len(completed)
        state["completed_bytes"] = sum(int(v["size"]) for v in completed.values())
        state["percent"] = 100.0 * state["completed_bytes"] / max(1, state["total_bytes"])
        atomic_write_json(RUNTIME / "transfer_state.json", state)
        write_heartbeat(state)
        time.sleep(60)

    state.update({"status": "CORE_DOWNLOAD_COMPLETE", "completed_at": now_iso(), "current_phase": "COMPLETE"})
    atomic_write_json(RUNTIME / "transfer_state.json", state)
    write_heartbeat(state)
    marker = {
        "dataset_repo": args.repo_id,
        "dataset_revision": revision,
        "completed_time": now_iso(),
        "file_count": len(required),
        "total_bytes": total_bytes(required),
        "transfer_manifest_path": str(args.manifest),
    }
    marker_path = "/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/runtime/CORE_DOWNLOAD_COMPLETE"
    remote_cmd = f"mkdir -p {shquote(str(Path(marker_path).parent))} && cat > {shquote(marker_path)} <<'EOF'\n{json.dumps(marker, indent=2, sort_keys=True)}\nEOF\nsync"
    subprocess.run(ssh_cmd(args.pai_key, args.pai_host, remote_cmd), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
