#!/usr/bin/env python3
"""Selectively extract VOR-Train/BG videos needed by Exp26 Gate64.

The Gate64 protocol manifest stores archive member paths rather than local
files. This script streams the split VOR-Train tar.gz archive and writes only
the exact `winner_member_path` files requested by the locked manifest. It never
extracts the full archive.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import shutil
import tarfile
from pathlib import Path


DEFAULT_REVISION = "fa09dc61128ca0418a4a13364d97a08018ea9cc7"
DEFAULT_ARCHIVE_DIR = (
    Path("/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/downloads") / DEFAULT_REVISION
)


def natural_key(name: str) -> list[object]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def normalize_member_name(name: str) -> str:
    return name.replace("\\", "/").lstrip("./")


def unsafe_member_reason(member: tarfile.TarInfo) -> str:
    raw_name = member.name.replace("\\", "/")
    if raw_name.startswith("/") or Path(raw_name).is_absolute():
        return "absolute_path"
    raw_parts = [p for p in raw_name.split("/") if p not in ("", ".")]
    if any(p == ".." for p in raw_parts):
        return "path_traversal"
    if member.issym() or member.islnk():
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class MultiPartReader(io.RawIOBase):
    def __init__(self, parts: list[Path]):
        super().__init__()
        if not parts:
            raise ValueError("No VOR-Train archive parts found")
        self.parts = parts
        self.index = 0
        self.current = parts[0].open("rb")
        self.exhausted = False

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:  # noqa: ANN001 - RawIOBase protocol
        if self.exhausted:
            return 0
        view = memoryview(b)
        total = 0
        while total < len(view):
            n = self.current.readinto(view[total:])
            if n:
                total += n
                break
            self.current.close()
            self.index += 1
            if self.index >= len(self.parts):
                self.exhausted = True
                break
            self.current = self.parts[self.index].open("rb")
        return total

    def close(self) -> None:
        try:
            if not self.current.closed:
                self.current.close()
        finally:
            super().close()


def open_tar_stream(parts: list[Path]) -> tarfile.TarFile:
    raw = MultiPartReader(parts)
    buffered = io.BufferedReader(raw, buffer_size=8 * 1024 * 1024)
    gz = gzip.GzipFile(fileobj=buffered, mode="rb")
    return tarfile.open(fileobj=gz, mode="r|")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--output-manifest", type=Path, required=True)
    p.add_argument("--status-csv", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    targets = {normalize_member_name(str(row["winner_member_path"])) for row in rows}
    if not targets:
        raise ValueError("no winner_member_path targets")
    parts = sorted(args.archive_dir.glob("VOR-Train.tar.gz.part_*"), key=lambda p: natural_key(p.name))
    args.output_root.mkdir(parents=True, exist_ok=True)
    found: dict[str, Path] = {}
    status_rows: list[dict] = []
    with open_tar_stream(parts) as tar:
        for idx, member in enumerate(tar):
            member_name = normalize_member_name(member.name)
            if member_name not in targets:
                continue
            reason = unsafe_member_reason(member)
            if reason:
                status_rows.append(
                    {
                        "member_path": member_name,
                        "status": "UNSAFE",
                        "size": int(member.size or 0),
                        "output_path": "",
                        "sha256": "",
                        "reason": reason,
                    }
                )
                continue
            dest = safe_output_path(args.output_root, member_name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if member.isfile():
                src = tar.extractfile(member)
                if src is None:
                    status = "NO_FILEOBJ"
                    digest = ""
                else:
                    tmp = dest.with_suffix(dest.suffix + ".tmp")
                    with tmp.open("wb") as f:
                        shutil.copyfileobj(src, f, length=8 * 1024 * 1024)
                    tmp.replace(dest)
                    digest = sha256_file(dest)
                    found[member_name] = dest
                    status = "OK"
            else:
                status = "NOT_FILE"
                digest = ""
            status_rows.append(
                {
                    "member_path": member_name,
                    "status": status,
                    "size": int(member.size or 0),
                    "output_path": str(dest) if dest.exists() else "",
                    "sha256": digest,
                    "reason": "",
                }
            )
            if len(found) == len(targets):
                break
    by_member = {normalize_member_name(row["winner_member_path"]): row for row in rows}
    output_rows: list[dict] = []
    for member_name, row in by_member.items():
        path = found.get(member_name)
        if path is None:
            status_rows.append(
                {
                    "member_path": member_name,
                    "status": "MISSING",
                    "size": "",
                    "output_path": "",
                    "sha256": "",
                    "reason": "target member was not found in archive stream",
                }
            )
            continue
        out_row = dict(row)
        out_row.update({"local_video_path": str(path), "source_video_path": str(path), "status": "GATE64_BG_EXTRACTED"})
        output_rows.append(out_row)
    write_jsonl(args.output_manifest, output_rows)
    args.status_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["member_path", "status", "size", "output_path", "sha256", "reason"]
    with args.status_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in fields} for row in status_rows])
    summary = {
        "targets": len(targets),
        "found": len(found),
        "missing": len(targets) - len(found),
        "manifest": str(args.output_manifest),
        "status_csv": str(args.status_csv),
        "archive_dir": str(args.archive_dir),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["missing"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
