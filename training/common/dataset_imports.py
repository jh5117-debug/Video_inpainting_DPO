"""Robust imports for legacy top-level ``dataset`` helpers.

Some environments can have a third-party ``dataset`` package on sys.path.  The
training scripts only need two small legacy helpers, so fall back to loading
them directly from this repository when the normal package import is shadowed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def import_dataset_file_helpers(project_root):
    """Return ``FileClient`` and ``imfrombytes`` from the repo dataset helpers."""

    root = Path(project_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        from dataset.file_client import FileClient
        from dataset.img_util import imfrombytes

        return FileClient, imfrombytes
    except ModuleNotFoundError as exc:
        if exc.name not in {"dataset", "dataset.file_client", "dataset.img_util"}:
            raise

    file_client_mod = _load_module_from_path(
        "_repo_dataset_file_client",
        root / "dataset" / "file_client.py",
    )
    img_util_mod = _load_module_from_path(
        "_repo_dataset_img_util",
        root / "dataset" / "img_util.py",
    )
    return file_client_mod.FileClient, img_util_mod.imfrombytes
