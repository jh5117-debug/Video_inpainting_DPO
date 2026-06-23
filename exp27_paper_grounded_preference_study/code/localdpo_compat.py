"""Exp27 compatibility wrapper for LocalDPO random mask generation.

The official cached LocalDPO commit is not edited. In the current matplotlib
version, Agg canvas exposes ARGB/RGBA buffers, while the official code reshapes
that buffer as RGB. This wrapper patches only the canvas conversion function
used by `get_random_shape`, preserving the rest of the official mask/motion
logic.
"""

from __future__ import annotations

import importlib.util
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import cv2


PAPER_CODE_ROOT = Path(os.environ.get("EXP27_PAPER_CODE_ROOT", "/home/hj/video_dpo_paper_code_cache/repos"))
LOCALDPO_RANDOM_MASK = PAPER_CODE_ROOT / "Local-DPO" / "innerT2V" / "utils" / "random_mask_gen.py"


def load_official_random_mask_module():
    root = LOCALDPO_RANDOM_MASK.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("official_localdpo_random_mask_gen_compat", LOCALDPO_RANDOM_MASK)
    if spec is None or spec.loader is None:
        raise ImportError(LOCALDPO_RANDOM_MASK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canvas_argb_to_rgb_array(fig) -> np.ndarray:
    """Return an RGB array from a matplotlib Agg canvas.

    Official LocalDPO calls `tostring_argb()` and then reshapes as 3-channel RGB.
    In matplotlib 3.10 this buffer is 4-channel ARGB. Dropping the alpha channel
    restores the likely intended RGB semantics.
    """

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    expected_argb = width * height * 4
    if argb.size != expected_argb:
        raise ValueError(f"unexpected canvas buffer size {argb.size}; expected {expected_argb}")
    return argb.reshape((height, width, 4))[:, :, 1:4]


def patch_get_random_shape(module):
    """Patch only LocalDPO's matplotlib canvas conversion."""

    def get_random_shape_compat(edge_num=9, ratio=0.7, width=432, height=240):
        points_num = edge_num * 3 + 1
        angles = np.linspace(0, 2 * np.pi, points_num)
        codes = np.full(points_num, module.Path.CURVE4)
        codes[0] = module.Path.MOVETO
        verts = np.stack((np.cos(angles), np.sin(angles))).T * (
            2 * ratio * np.random.random(points_num) + 1 - ratio
        )[:, None]
        verts[-1, :] = verts[0, :]
        path = module.Path(verts, codes)
        fig = module.plt.figure()
        ax = fig.add_subplot(111)
        patch = module.patches.PathPatch(path, facecolor="black", lw=2)
        ax.add_patch(patch)
        ax.set_xlim(np.min(verts) * 1.1, np.max(verts) * 1.1)
        ax.set_ylim(np.min(verts) * 1.1, np.max(verts) * 1.1)
        ax.axis("off")
        data = canvas_argb_to_rgb_array(fig)
        module.plt.close(fig)
        data = cv2.resize(data, (width, height))[:, :, 0]
        data = (1 - np.array(data > 0).astype(np.uint8)) * 255
        coordinates = np.where(data > 0)
        xmin, xmax = np.min(coordinates[0]), np.max(coordinates[0])
        ymin, ymax = np.min(coordinates[1]), np.max(coordinates[1])
        return Image.fromarray(data).crop((ymin, xmin, ymax, xmax))

    module.get_random_shape = get_random_shape_compat
    return module


def localdpo_mask_digest_compat(
    *,
    seed: int,
    video_length: int = 13,
    image_height: int = 120,
    image_width: int = 216,
    connected_components: int = 1,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    try:
        module = patch_get_random_shape(load_official_random_mask_module())
    except FileNotFoundError as exc:
        return {
            "status": "blocked_official_code_missing",
            "error": repr(exc),
            "source": str(LOCALDPO_RANDOM_MASK),
        }
    if connected_components == 1:
        masks = module.create_random_shape_with_random_motion(
            video_length,
            zoomin=0.9,
            zoomout=1.1,
            rotmin=1,
            rotmax=10,
            imageHeight=image_height,
            imageWidth=image_width,
        )
    else:
        masks = module.create_random_shape_with_random_motion_multiple_connected_components(
            video_length,
            zoomin=0.9,
            zoomout=1.1,
            rotmin=1,
            rotmax=10,
            cc_ratio=max(1, connected_components),
            fix_area=0,
            imageHeight=image_height,
            imageWidth=image_width,
        )
    arr = np.stack([np.array(m, dtype=np.uint8) for m in masks], axis=0)
    import hashlib

    return {
        "status": "passed_with_official_code_compatibility_patch",
        "shape": list(arr.shape),
        "sum": int(arr.sum()),
        "mean": float(arr.mean()),
        "first_frame_sum": int(arr[0].sum()),
        "last_frame_sum": int(arr[-1].sum()),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        "patch": "drop ARGB alpha channel before official RGB postprocess; provide cv2 dependency in wrapper",
        "source": str(LOCALDPO_RANDOM_MASK),
    }
