#!/usr/bin/env python3
"""Shared helpers for Exp25 EffectErase VOR transfer."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


REPO_ID = "FudanCVL/EffectErase"
REPO_TYPE = "dataset"
EXP_ROOT = Path("exp25_vor_or_preference_data")
REPORTS = Path("reports")
RUNTIME = EXP_ROOT / "runtime"
LOGS = EXP_ROOT / "logs"
REGISTRY = Path("experiment_registry/exp25_vor_or_preference_data")

DEFAULT_HF_ENV = Path("/home/hj/.venvs/hf_effecterase")
DEFAULT_HF_AUTH_HOME = Path("/home/hj/.cache/huggingface_effecterase_auth")
DEFAULT_HAL_STAGING = Path("/home/hj/exp25_effecterase_staging")
DEFAULT_PAI_HOST = "root@47.103.26.60"
DEFAULT_PAI_KEY = Path("/home/hj/.ssh/codex_pai")
DEFAULT_PAI_DOWNLOAD_DIR = Path(
    "/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/downloads"
)


def ensure_dirs() -> None:
    for path in [EXP_ROOT, REPORTS, RUNTIME, LOGS, REGISTRY]:
        path.mkdir(parents=True, exist_ok=True)


def run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None,
        check: bool = True, stdout=None, stderr=None, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, env=env, cwd=str(cwd) if cwd else None, check=check,
                          stdout=stdout, stderr=stderr, text=text)


def ssh_cmd(key: Path, host: str, remote_cmd: str) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=30",
        "-i",
        str(key),
        host,
        remote_cmd,
    ]


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def append_csv(path: Path, fieldnames: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def safe_job_name(filename: str) -> str:
    cleaned = filename.strip("/").replace("/", "__")
    return re.sub(r"[^A-Za-z0-9._+=@-]+", "_", cleaned)


def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def required_group(filename: str) -> str | None:
    base = filename.split("/")[-1]
    if filename == "README.md" or base == "README.md":
        return "README"
    for prefix, group in [
        ("VOR-Eval.tar.gz.part_", "VOR-Eval"),
        ("VOR-Train-MASK.tar.gz.part_", "VOR-Train-MASK"),
        ("VOR-Train.tar.gz.part_", "VOR-Train"),
    ]:
        if base.startswith(prefix):
            return group
    return None


def required_inventory(files: Iterable[dict]) -> list[dict]:
    selected = []
    for item in files:
        group = required_group(item["filename"])
        if group:
            new = dict(item)
            new["group"] = group
            new["required"] = True
            selected.append(new)
    order = {"README": 0, "VOR-Eval": 1, "VOR-Train-MASK": 2, "VOR-Train": 3}
    selected.sort(key=lambda x: (order[x["group"]], natural_key(x["filename"])))
    return selected


def part_token(filename: str) -> str | None:
    m = re.search(r"\.part_([^/]+)$", filename)
    return m.group(1) if m else None


def continuity_report(files: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for group in ["VOR-Eval", "VOR-Train-MASK", "VOR-Train"]:
        group_files = [f for f in files if f.get("group") == group]
        tokens = [part_token(f["filename"]) for f in group_files]
        tokens = [t for t in tokens if t is not None]
        numeric = []
        non_numeric = []
        for t in tokens:
            if re.fullmatch(r"\d+", t):
                numeric.append(int(t))
            else:
                non_numeric.append(t)
        missing: list[int] = []
        contiguous = True
        if numeric and not non_numeric:
            lo, hi = min(numeric), max(numeric)
            present = set(numeric)
            missing = [i for i in range(lo, hi + 1) if i not in present]
            contiguous = not missing and len(present) == len(numeric)
        elif tokens:
            contiguous = len(tokens) == len(set(tokens))
        out[group] = {
            "count": len(group_files),
            "tokens": tokens,
            "numeric": bool(numeric and not non_numeric),
            "contiguous": contiguous,
            "missing": missing,
            "duplicates": len(tokens) - len(set(tokens)),
        }
    return out


def shquote(s: str | Path) -> str:
    return "'" + str(s).replace("'", "'\"'\"'") + "'"


def realpath_under(path: Path, parent: Path) -> bool:
    try:
        p = path.resolve()
        root = parent.resolve()
    except FileNotFoundError:
        p = path.parent.resolve() / path.name
        root = parent.resolve()
    return p == root or root in p.parents


def safe_remove_tree(path: Path, allowed_parent: Path) -> int:
    path = path.resolve()
    allowed_parent = allowed_parent.resolve()
    if not realpath_under(path, allowed_parent):
        raise RuntimeError(f"Refusing cleanup outside allowed parent: {path} not under {allowed_parent}")
    if path in [Path("/"), Path("/home/hj"), DEFAULT_HF_AUTH_HOME.resolve()]:
        raise RuntimeError(f"Refusing unsafe cleanup target: {path}")
    size = dir_size(path) if path.exists() else 0
    shutil.rmtree(path)
    return size


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def total_bytes(files: Iterable[dict]) -> int:
    return sum(int(f.get("size") or 0) for f in files)


@dataclass
class FileState:
    filename: str
    group: str
    size: int
    status: str = "PENDING"
    hal_sha256: str = ""
    pai_sha256: str = ""
    started_at: str = ""
    completed_at: str = ""
    retries: int = 0
    last_error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")

