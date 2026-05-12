#!/usr/bin/env python
"""Prepare the released VideoDPO VC2 preference dataset on a shared filesystem.

The original VideoDPO dataloader is happiest when the META entries in
train_data.yaml are absolute paths.  This helper downloads the HF dataset, finds
the directory containing metadata.json/pair.json, and writes an absolute
train_data.yaml for SC/H20-style jobs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import tarfile
import zipfile

import yaml


VIDEO_EXTS = {".avi", ".gif", ".mp4", ".webm"}


def _find_dataset_root(root: Path) -> Path:
    candidates = []
    if (root / "metadata.json").is_file() and (root / "pair.json").is_file():
        candidates.append(root)
    candidates.extend(
        p for p in root.rglob("metadata.json")
        if (p.parent / "pair.json").is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a directory containing metadata.json and pair.json under {root}"
        )
    candidates = sorted({p if p.is_dir() else p.parent for p in candidates}, key=lambda p: (len(p.parts), str(p)))
    return candidates[0]


def _copy_or_link(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(src)
    elif mode == "copy":
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _archive_paths(root: Path) -> list[Path]:
    suffixes = {
        ".tar",
        ".tgz",
        ".zip",
        ".gz",
        ".bz2",
        ".xz",
    }
    archives = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.endswith((".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".zip")) or path.suffix.lower() in suffixes:
            archives.append(path)
    return sorted(archives, key=lambda p: (len(p.parts), str(p)))


def _extract_archives(root: Path, extract_dir: Path) -> None:
    archives = _archive_paths(root)
    if not archives:
        return
    extract_dir.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        marker = extract_dir / f".extracted_{archive.name.replace('/', '_')}"
        if marker.exists():
            continue
        print(f"[prepare-vc2] extracting archive={archive} to={extract_dir}")
        if tarfile.is_tarfile(archive):
            with tarfile.open(archive) as tf:
                tf.extractall(extract_dir)
        elif zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(extract_dir)
        else:
            print(f"[prepare-vc2][warn] unsupported archive, skipping: {archive}")
            continue
        marker.write_text(str(archive), encoding="utf-8")


def _path_suffixes(path: Path | PurePosixPath) -> list[str]:
    parts = [p for p in path.parts if p not in ("", "/")]
    suffixes: list[str] = []
    for marker in ("dataset", "vidpro10k-vc2-dataset", "text2video2-10k"):
        if marker in parts:
            idx = parts.index(marker)
            starts = [idx]
            if marker in {"dataset", "vidpro10k-vc2-dataset"}:
                starts.append(idx + 1)
            for start in starts:
                if start < len(parts):
                    suffixes.append("/".join(parts[start:]))
    for n in range(min(10, len(parts)), 0, -1):
        suffixes.append("/".join(parts[-n:]))
    seen = set()
    out = []
    for suffix in suffixes:
        if suffix and suffix not in seen:
            seen.add(suffix)
            out.append(suffix)
    return out


def _build_video_index(search_roots: list[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    seen_files = set()
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
                continue
            resolved = path.resolve()
            if resolved in seen_files:
                continue
            seen_files.add(resolved)
            for suffix in _path_suffixes(path):
                index.setdefault(suffix, []).append(resolved)
    return index


def _find_local_video(clip_path: str, dataset_root: Path, video_index: dict[str, list[Path]]) -> Path | None:
    raw = PurePosixPath(clip_path)
    if raw.is_absolute() and Path(clip_path).is_file():
        return Path(clip_path).resolve()
    local = dataset_root / clip_path
    if local.is_file():
        return local.resolve()

    # A previous bad rewrite may have produced an absolute path under
    # dataset_root that does not exist.  Recover by also trying the path
    # relative to dataset_root before suffix matching.
    try:
        rel_to_dataset = Path(clip_path).relative_to(dataset_root)
    except ValueError:
        rel_to_dataset = None
    if rel_to_dataset is not None:
        local_rel = dataset_root / rel_to_dataset
        if local_rel.is_file():
            return local_rel.resolve()
        for suffix in _path_suffixes(PurePosixPath(str(rel_to_dataset))):
            matches = video_index.get(suffix, [])
            if len(matches) == 1:
                return matches[0]

    for suffix in _path_suffixes(raw):
        matches = video_index.get(suffix, [])
        if len(matches) == 1:
            return matches[0]
    return None


def _rewrite_metadata_clip_paths(dataset_root: Path, search_roots: list[Path]) -> tuple[int, int]:
    metadata_path = dataset_root / "metadata.json"
    if not metadata_path.is_file():
        return 0, 0

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if not isinstance(metadata, list):
        return 0, 0

    video_index = _build_video_index(search_roots)
    changed = 0
    unresolved = 0
    for item in metadata:
        basic = item.get("basic") if isinstance(item, dict) else None
        if not isinstance(basic, dict) or "clip_path" not in basic:
            continue
        clip_path = str(basic["clip_path"])
        local_video = _find_local_video(clip_path, dataset_root, video_index)
        if local_video is None:
            unresolved += 1
            continue
        fixed_path = str(local_video)
        if fixed_path != clip_path:
            basic["clip_path"] = fixed_path
            changed += 1

    if changed:
        backup = dataset_root / "metadata.json.original_paths.bak"
        if not backup.exists():
            shutil.copy2(metadata_path, backup)
        tmp = dataset_root / "metadata.json.tmp"
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False)
        tmp.replace(metadata_path)
    return changed, unresolved


def _resolve_for_videodpo(dataset_root: Path, clip_path: str) -> Path:
    path = Path(clip_path)
    if path.is_absolute():
        return path
    return dataset_root / clip_path


def _validate_metadata_clips(dataset_root: Path, limit: int = 20) -> tuple[str, Path, bool, int]:
    metadata_path = dataset_root / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if not isinstance(metadata, list) or not metadata:
        raise RuntimeError(f"metadata.json has no list entries: {metadata_path}")

    first_clip = str(metadata[0]["basic"]["clip_path"])
    first_resolved = _resolve_for_videodpo(dataset_root, first_clip)
    first_exists = first_resolved.is_file()
    missing = 0
    for item in metadata[:limit]:
        clip_path = str(item["basic"]["clip_path"])
        if not _resolve_for_videodpo(dataset_root, clip_path).is_file():
            missing += 1
    return first_clip, first_resolved, first_exists, missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="JiaHuang01/vidpro10k-vc2-dataset")
    parser.add_argument("--target_root", required=True,
                        help="Final directory, e.g. ${PROJECT_DATA}/VideoDPO/data/vidpro-vc2-dpo-dataset")
    parser.add_argument("--output_yaml", required=True,
                        help="Absolute train_data.yaml consumed by VideoDPO")
    parser.add_argument("--download_dir", default=None,
                        help="Optional HF snapshot download directory; defaults to target_root")
    parser.add_argument("--link_mode", choices=["symlink", "copy"], default="copy",
                        help="How to place a nested downloaded dataset at target_root")
    parser.add_argument("--skip_download", action="store_true",
                        help="Use an already existing target_root/download_dir")
    args = parser.parse_args()

    target_root = Path(args.target_root).expanduser().resolve()
    output_yaml = Path(args.output_yaml).expanduser().resolve()
    download_dir = Path(args.download_dir).expanduser().resolve() if args.download_dir else target_root

    if not args.skip_download:
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:  # pragma: no cover - depends on cluster env
            raise RuntimeError(
                "huggingface_hub is required for download. Install it in the active env "
                "or rerun with --skip_download after manually placing the dataset."
            ) from exc
        download_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(download_dir),
            local_dir_use_symlinks=False,
        )

    try:
        source_root = _find_dataset_root(download_dir)
    except FileNotFoundError:
        _extract_archives(download_dir, download_dir / "_extracted")
        source_root = _find_dataset_root(download_dir)
    if source_root.resolve() != target_root:
        if target_root.exists() and not (
            (target_root / "metadata.json").is_file()
            and (target_root / "pair.json").is_file()
        ):
            # Hugging Face snapshots may put archive files directly in target_root;
            # after extraction, source_root can still be copied into this directory.
            has_archives_only = bool(_archive_paths(target_root)) or (target_root / "_extracted").exists()
            if not has_archives_only:
                raise FileExistsError(f"target_root exists but is not a VideoDPO dataset root: {target_root}")
        if target_root.exists() and any(target_root.iterdir()) and source_root.is_relative_to(target_root):
            pass
        elif target_root.exists() and not ((target_root / "metadata.json").is_file() and (target_root / "pair.json").is_file()):
            raise FileExistsError(f"target_root exists but is not a VideoDPO dataset root: {target_root}")
        else:
            _copy_or_link(source_root, target_root, args.link_mode)

    dataset_root = _find_dataset_root(target_root).resolve()
    changed, unresolved = _rewrite_metadata_clip_paths(
        dataset_root,
        sorted({target_root, download_dir, dataset_root}, key=lambda p: str(p)),
    )
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with output_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"META": [str(dataset_root)]}, f, sort_keys=False)
    first_clip, first_resolved, first_exists, missing_sample = _validate_metadata_clips(dataset_root)

    print(f"[prepare-vc2] repo_id={args.repo_id}")
    print(f"[prepare-vc2] dataset_root={dataset_root}")
    print(f"[prepare-vc2] output_yaml={output_yaml}")
    print(f"[prepare-vc2] rewritten_clip_paths={changed} unresolved_clip_paths={unresolved}")
    print(f"[prepare-vc2] first_clip_path={first_clip}")
    print(f"[prepare-vc2] first_clip_resolved={first_resolved}")
    print(f"[prepare-vc2] first_clip_exists={'yes' if first_exists else 'no'}")
    print(f"[prepare-vc2] sample_missing_clip_paths={missing_sample}/20")
    if unresolved:
        raise RuntimeError(
            f"{unresolved} clip_path entries could not be resolved under {target_root}. "
            "Inspect metadata.json and the extracted archive layout."
        )
    if not first_exists:
        raise RuntimeError(
            f"First clip_path does not exist after rewrite: {first_resolved}. "
            "The dataset prepare step and health check would disagree without this hard failure."
        )
    if missing_sample:
        raise RuntimeError(
            f"{missing_sample}/20 sampled clip_path entries are missing after rewrite. "
            "The metadata.json file is still not safe for VideoDPO training."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
