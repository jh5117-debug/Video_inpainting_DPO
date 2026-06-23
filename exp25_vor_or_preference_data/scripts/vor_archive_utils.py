#!/usr/bin/env python3
"""Archive utilities for Exp25 EffectErase VOR.

The VOR archives are split gzip-compressed tar streams.  These helpers keep
the default operations lightweight, while still supporting resumable full
stream scans and selective extraction when explicitly requested on PAI.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import os
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


GROUP_PREFIX = {
    "VOR-Eval": "VOR-Eval.tar.gz.part_",
    "VOR-Train-MASK": "VOR-Train-MASK.tar.gz.part_",
    "VOR-Train": "VOR-Train.tar.gz.part_",
}

DEFAULT_REVISION = "fa09dc61128ca0418a4a13364d97a08018ea9cc7"
DEFAULT_PAI_ARCHIVE_ROOT = Path(
    "/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/downloads"
) / DEFAULT_REVISION


def natural_key(name: str) -> list[object]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def atomic_write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def append_csv(path: Path, fieldnames: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def part_token(path: Path | str) -> str | None:
    m = re.search(r"\.part_([^/]+)$", str(path))
    return m.group(1) if m else None


def group_for_name(name: str) -> str | None:
    base = Path(name).name
    for group, prefix in GROUP_PREFIX.items():
        if base.startswith(prefix):
            return group
    if base == "README.md":
        return "README"
    return None


def group_parts(archive_dir: Path, group: str) -> list[Path]:
    prefix = GROUP_PREFIX[group]
    return sorted(archive_dir.glob(prefix + "*"), key=lambda p: natural_key(p.name))


def continuity_for_parts(parts: Iterable[Path]) -> dict:
    parts = list(parts)
    tokens = [part_token(p.name) for p in parts]
    tokens = [t for t in tokens if t is not None]
    numeric: list[int] = []
    non_numeric: list[str] = []
    for token in tokens:
        if re.fullmatch(r"\d+", token):
            numeric.append(int(token))
        else:
            non_numeric.append(token)
    missing: list[int] = []
    contiguous = True
    if numeric and not non_numeric:
        lo, hi = min(numeric), max(numeric)
        present = set(numeric)
        missing = [i for i in range(lo, hi + 1) if i not in present]
        contiguous = not missing and len(present) == len(numeric)
    elif tokens:
        contiguous = len(tokens) == len(set(tokens))
    return {
        "count": len(parts),
        "tokens": tokens,
        "numeric": bool(numeric and not non_numeric),
        "contiguous": contiguous,
        "missing": missing,
        "duplicates": len(tokens) - len(set(tokens)),
        "total_bytes": sum(p.stat().st_size for p in parts if p.exists()),
    }


def normalize_member_name(name: str) -> str:
    return name.replace("\\", "/").lstrip("./")


def unsafe_member_reason(member: tarfile.TarInfo) -> str:
    raw_name = member.name.replace("\\", "/")
    if raw_name.startswith("/") or Path(raw_name).is_absolute():
        return "absolute_path"
    raw_parts = [p for p in raw_name.split("/") if p not in ("", ".")]
    if any(p == ".." for p in raw_parts):
        return "path_traversal"
    name = normalize_member_name(member.name)
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if not parts:
        return "empty_path"
    if member.issym() or member.islnk():
        target = member.linkname or ""
        if target.startswith("/") or ".." in Path(target).parts:
            return "unsafe_link"
        return "link_member"
    if member.isdev():
        return "device_member"
    return ""


def safe_output_path(output_root: Path, member_name: str) -> Path:
    normalized = normalize_member_name(member_name)
    out = (output_root / normalized).resolve()
    root = output_root.resolve()
    if out != root and root not in out.parents:
        raise ValueError(f"Refusing archive path outside output root: {member_name}")
    return out


def sample_id_from_member(name: str) -> str:
    """Best-effort sample id from an archive member path.

    The script records the raw member path too; this heuristic is only used for
    selective extraction and split-overlap guards until the true metadata index
    is available.
    """

    name = normalize_member_name(name)
    parts = [p for p in name.split("/") if p]
    while parts and parts[0] in {"VOR-Train", "VOR-Train-MASK", "VOR-Eval", "train", "eval", "mask", "masks"}:
        parts = parts[1:]
    if not parts:
        return ""
    if len(parts) >= 2 and re.fullmatch(r"\d{1,6}", parts[0]):
        return f"{parts[0]}/{parts[1]}"
    stem = Path(parts[0]).stem
    return stem


class MultiPartReader(io.RawIOBase):
    """Sequential read-only file object over split archive parts."""

    def __init__(self, parts: list[Path]):
        super().__init__()
        if not parts:
            raise ValueError("No archive parts provided")
        self.parts = parts
        self.index = 0
        self.current = parts[0].open("rb")
        self.offset = 0

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:  # noqa: ANN001 - RawIOBase protocol
        view = memoryview(b)
        total = 0
        while total < len(view):
            n = self.current.readinto(view[total:])
            if n:
                total += n
                self.offset += n
                break
            self.current.close()
            self.index += 1
            if self.index >= len(self.parts):
                break
            self.current = self.parts[self.index].open("rb")
        return total

    def close(self) -> None:
        try:
            self.current.close()
        finally:
            super().close()


def open_tar_stream(parts: list[Path]) -> tarfile.TarFile:
    raw = MultiPartReader(parts)
    buffered = io.BufferedReader(raw, buffer_size=8 * 1024 * 1024)
    gz = gzip.GzipFile(fileobj=buffered, mode="rb")
    return tarfile.open(fileobj=gz, mode="r|")


@dataclass
class MemberRecord:
    group: str
    member_index: int
    member_path: str
    sample_id: str
    type: str
    size: int
    mtime: int
    unsafe_reason: str

    @classmethod
    def from_tarinfo(cls, group: str, idx: int, member: tarfile.TarInfo) -> "MemberRecord":
        if member.isdir():
            typ = "dir"
        elif member.isfile():
            typ = "file"
        elif member.issym():
            typ = "symlink"
        elif member.islnk():
            typ = "hardlink"
        else:
            typ = "other"
        return cls(
            group=group,
            member_index=idx,
            member_path=normalize_member_name(member.name),
            sample_id=sample_id_from_member(member.name),
            type=typ,
            size=int(member.size or 0),
            mtime=int(member.mtime or 0),
            unsafe_reason=unsafe_member_reason(member),
        )

    def to_dict(self) -> dict:
        return {
            "group": self.group,
            "member_index": self.member_index,
            "member_path": self.member_path,
            "sample_id": self.sample_id,
            "type": self.type,
            "size": self.size,
            "mtime": self.mtime,
            "unsafe_reason": self.unsafe_reason,
        }
