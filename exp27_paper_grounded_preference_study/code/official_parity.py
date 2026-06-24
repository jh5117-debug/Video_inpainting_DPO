"""Small exact-parity helpers for paper-grounded Exp27 gates.

These functions intentionally mirror official public code in:
- AIDC-AI/Diffusion-SDPO train.py
- Whynot0101/Linear-DPO train/train_sd_dpo.py and train/train_sd3_dpo.py
- 1170300714/Local-DPO innerT2V/utils/random_mask_gen.py

The helpers are CPU-safe and do not import project trainers.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def _resolve_paper_code_root() -> Path:
    env = os.environ.get("EXP27_PAPER_CODE_ROOT")
    if env:
        return Path(env)
    candidates = [
        Path("/home/hj/video_dpo_paper_code_cache/repos"),
        Path("/home/hj/video_dpo_paper_code_cache"),
        Path("/mnt/nas/hj/video_dpo_paper_code_cache"),
    ]
    for root in candidates:
        if (root / "Local-DPO" / "innerT2V" / "utils" / "random_mask_gen.py").exists():
            return root
        matches = list(root.glob("Local-DPO_*")) if root.exists() else []
        for match in matches:
            if (match / "innerT2V" / "utils" / "random_mask_gen.py").exists():
                return root
    return Path("/home/hj/video_dpo_paper_code_cache/repos")


PAPER_CODE_ROOT = _resolve_paper_code_root()


def install_localdpo_matplotlib_rgb_shim() -> str:
    """Patch current matplotlib to match Local-DPO's intended RGB buffer.

    The pinned Local-DPO code calls ``fig.canvas.tostring_argb()`` and then
    reshapes the returned bytes as ``H x W x 3``.  Modern Matplotlib correctly
    returns four-channel ARGB bytes for that method, so the official script
    raises a reshape error before any mask can be generated.  We keep the
    official file read-only and install a narrow runtime shim that returns RGB
    bytes from ``buffer_rgba()`` for this call.
    """

    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except Exception as exc:  # pragma: no cover - only used in env diagnostics.
        return f"shim_unavailable:{exc!r}"

    if getattr(FigureCanvasAgg, "_exp27_localdpo_rgb_shim", False):
        return "already_installed"

    original = FigureCanvasAgg.tostring_argb

    def _rgb_bytes(self):  # type: ignore[no-untyped-def]
        rgba = np.asarray(self.buffer_rgba())
        return rgba[:, :, :3].tobytes()

    FigureCanvasAgg._exp27_original_tostring_argb = original  # type: ignore[attr-defined]
    FigureCanvasAgg.tostring_argb = _rgb_bytes  # type: ignore[assignment]
    FigureCanvasAgg._exp27_localdpo_rgb_shim = True  # type: ignore[attr-defined]
    return "installed"


def exp27_sdpo_safe_lambda(
    pred_eps: torch.Tensor,
    target_eps: torch.Tensor,
    mu: float,
    eps: float = 1e-9,
    max_lambda: float = 1.0,
) -> torch.Tensor:
    """Mirror Diffusion-SDPO official output-space safe lambda."""
    if not 0.0 <= mu <= 1.0:
        raise ValueError("mu should be within [0,1]")
    pred_w, pred_l = pred_eps.detach().requires_grad_(True).chunk(2, dim=0)
    target_w, target_l = target_eps.detach().chunk(2, dim=0)
    winner_proxy = ((pred_w - target_w) ** 2).mean()
    loser_proxy = ((pred_l - target_l) ** 2).mean()
    grad_w = torch.autograd.grad(winner_proxy, pred_w, create_graph=False, retain_graph=False)[0]
    grad_l = torch.autograd.grad(loser_proxy, pred_l, create_graph=False, retain_graph=False)[0]
    num = grad_w.flatten().pow(2).sum()
    den = (grad_l.flatten() * grad_w.flatten()).sum()
    lam_cap = ((1.0 - float(mu)) * num) / (den + eps)
    lam = torch.where(
        den > 0,
        lam_cap,
        torch.tensor(max_lambda, device=pred_eps.device, dtype=pred_eps.dtype),
    )
    return lam.clamp(min=0.0, max=max_lambda).detach().to(dtype=pred_eps.dtype)


def load_official_sdpo_lambda():
    path = PAPER_CODE_ROOT / "Diffusion-SDPO" / "train.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    target = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_adaptive_lose_l_scale":
            target = node
            break
    if target is None:
        raise ImportError(f"get_adaptive_lose_l_scale not found in {path}")
    module = ast.Module(body=[target], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - exact official function extraction.
    return namespace["get_adaptive_lose_l_scale"]


def linear_dpo_clip_ratio(model_diff: torch.Tensor, ref_diff: torch.Tensor, beta_dpo: float, eta_dpo: float) -> torch.Tensor:
    """Mirror Linear-DPO official ratio = 0.2 * beta * gap + 0.5 then clamp."""
    with torch.no_grad():
        ratio = 0.2 * beta_dpo * (model_diff - ref_diff) + 0.5
        return torch.clamp(ratio, min=eta_dpo, max=1.0 - eta_dpo)


def linear_dpo_loss(
    model_losses_w: torch.Tensor,
    model_losses_l: torch.Tensor,
    ref_losses_w: torch.Tensor,
    ref_losses_l: torch.Tensor,
    beta_dpo: float,
    eta_dpo: float,
) -> torch.Tensor:
    model_diff = model_losses_w - model_losses_l
    ref_diff = ref_losses_w - ref_losses_l
    clip_ratio = linear_dpo_clip_ratio(model_diff, ref_diff, beta_dpo, eta_dpo)
    return (clip_ratio * (model_losses_w - model_losses_l)).mean()


def ema_update_tensor(ema: torch.Tensor, model: torch.Tensor, decay: float) -> torch.Tensor:
    """Mirror Linear-DPO ModelEMA parameter update."""
    return ema.mul(decay).add(model, alpha=1.0 - decay)


def load_localdpo_random_mask_module():
    localdpo_root = PAPER_CODE_ROOT / "Local-DPO"
    if not (localdpo_root / "innerT2V" / "utils" / "random_mask_gen.py").exists():
        matches = sorted(PAPER_CODE_ROOT.glob("Local-DPO_*"))
        for match in matches:
            if (match / "innerT2V" / "utils" / "random_mask_gen.py").exists():
                localdpo_root = match
                break
    root = localdpo_root / "innerT2V" / "utils"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    path = root / "random_mask_gen.py"
    spec = importlib.util.spec_from_file_location("official_localdpo_random_mask_gen", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "cv2"):
        import cv2  # type: ignore

        module.cv2 = cv2
    return module


def localdpo_mask_digest(
    *,
    seed: int,
    video_length: int = 13,
    image_height: int = 120,
    image_width: int = 216,
    connected_components: int = 1,
) -> dict:
    """Generate LocalDPO official masks and return deterministic digest metadata."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        shim_status = install_localdpo_matplotlib_rgb_shim()
        module = load_localdpo_random_mask_module()
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
    except FileNotFoundError as exc:
        return {
            "status": "blocked_official_code_missing",
            "error": repr(exc),
            "source": str(PAPER_CODE_ROOT / "Local-DPO" / "innerT2V" / "utils" / "random_mask_gen.py"),
        }
    except Exception as exc:  # noqa: BLE001 - parity gate records official runtime failures.
        return {
            "status": "blocked_official_code_runtime_error",
            "error": repr(exc),
            "source": str(PAPER_CODE_ROOT / "Local-DPO" / "innerT2V" / "utils" / "random_mask_gen.py"),
        }
    arr = np.stack([np.array(m, dtype=np.uint8) for m in masks], axis=0)
    return {
        "status": "passed",
        "shape": list(arr.shape),
        "sum": int(arr.sum()),
        "mean": float(arr.mean()),
        "first_frame_sum": int(arr[0].sum()),
        "last_frame_sum": int(arr[-1].sum()),
        "sha256": _sha256_bytes(arr.tobytes()),
        "matplotlib_rgb_shim": shim_status,
    }


def _sha256_bytes(data: bytes) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
