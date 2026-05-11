#!/usr/bin/env python
"""Prepare the released VideoDPO VC2 preference dataset on a shared filesystem.

The original VideoDPO dataloader is happiest when the META entries in
train_data.yaml are absolute paths.  This helper downloads the HF dataset, finds
the directory containing metadata.json/pair.json, and writes an absolute
train_data.yaml for SC/H20-style jobs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tarfile
import zipfile

import yaml


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
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with output_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"META": [str(dataset_root)]}, f, sort_keys=False)

    print(f"[prepare-vc2] repo_id={args.repo_id}")
    print(f"[prepare-vc2] dataset_root={dataset_root}")
    print(f"[prepare-vc2] output_yaml={output_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
