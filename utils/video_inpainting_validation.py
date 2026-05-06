"""PSNR/SSIM callback for VideoDPO runs on video-inpainting preference pairs."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_lightning.callbacks import Callback

from utils.callbacks import mainlogger


def _load_video_inpainting_metrics(repo_path: str | None = None):
    candidates = []
    if repo_path:
        candidates.append(Path(repo_path))
    if os.environ.get("VIDEOINPAINTING_REPO"):
        candidates.append(Path(os.environ["VIDEOINPAINTING_REPO"]))
    candidates.extend([
        Path("/home/nvme01/H20_Video_inpainting_DPO"),
        Path("/home/hj/Video_inpainting_DPO"),
    ])

    for root in candidates:
        metrics_path = root / "inference" / "metrics.py"
        if not metrics_path.exists():
            continue
        spec = importlib.util.spec_from_file_location("video_inpainting_metrics", metrics_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.compute_psnr, module.compute_ssim, str(metrics_path)

    raise ImportError(
        "Could not import compute_psnr/compute_ssim from a Video_inpainting_DPO checkout. "
        "Set VIDEOINPAINTING_REPO=/path/to/H20_Video_inpainting_DPO."
    )


def _video_tensor_to_uint8(video: torch.Tensor) -> list[np.ndarray]:
    """Convert [C,T,H,W] in [-1,1] to a list of uint8 RGB frames."""
    video = video.detach().float().cpu().clamp(-1.0, 1.0)
    video = video.add(1.0).mul(127.5).round().clamp(0, 255).byte()
    video = video.permute(1, 2, 3, 0).numpy()
    return [frame for frame in video]


def _resize_like(frame: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if frame.shape == reference.shape:
        return frame
    image = Image.fromarray(frame.astype(np.uint8))
    image = image.resize((reference.shape[1], reference.shape[0]), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


class VideoInpaintingMetricLogger(Callback):
    """
    Periodically samples the VideoDPO policy from the current DPO batch and compares
    the generated videos to the winner GT videos with the existing VideoInpainting
    PSNR/SSIM operators.

    This is a proxy metric: open-source VideoDPO is text-conditioned and does not
    consume masks/masked images, so it cannot run the DiffuEraser inpainting
    validation pipeline directly.
    """

    def __init__(
        self,
        every_n_train_steps=2000,
        max_videos=1,
        ddim_steps=25,
        unconditional_guidance_scale=12.0,
        video_inpainting_repo=None,
        enabled=True,
    ):
        super().__init__()
        self.every_n_train_steps = int(every_n_train_steps)
        self.max_videos = int(max_videos)
        self.ddim_steps = int(ddim_steps)
        self.unconditional_guidance_scale = float(unconditional_guidance_scale)
        self.video_inpainting_repo = video_inpainting_repo
        self.enabled = bool(enabled)
        self._last_step = -1

    def _should_run(self, trainer) -> bool:
        step = int(trainer.global_step)
        if not self.enabled or self.every_n_train_steps <= 0:
            return False
        if step <= 0 or step == self._last_step:
            return False
        if step % self.every_n_train_steps != 0:
            return False
        return int(getattr(trainer, "global_rank", 0)) == 0

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if not self._should_run(trainer):
            return

        step = int(trainer.global_step)
        self._last_step = step

        try:
            compute_psnr, compute_ssim, metrics_path = _load_video_inpainting_metrics(self.video_inpainting_repo)
        except Exception as exc:
            mainlogger.warning(f"[video_inpaint_val] step={step} skipped: {exc}")
            return

        was_training = pl_module.training
        pl_module.eval()
        try:
            logs = pl_module.log_images(
                batch,
                sample=True,
                ddim_steps=self.ddim_steps,
                unconditional_guidance_scale=self.unconditional_guidance_scale,
            )
            samples = logs.get("samples")
            if samples is None:
                mainlogger.warning(f"[video_inpaint_val] step={step} skipped: log_images returned no samples")
                return

            target = batch[pl_module.first_stage_key][:, :3]
            samples = samples.detach().cpu()
            target = target.detach().cpu()
            count = min(self.max_videos, target.shape[0], samples.shape[0])
            if count <= 0:
                mainlogger.warning(f"[video_inpaint_val] step={step} skipped: empty sample/target tensors")
                return

            per_video = []
            for i in range(count):
                pred_frames = _video_tensor_to_uint8(samples[i])
                gt_frames = _video_tensor_to_uint8(target[i])
                frame_psnr, frame_ssim = [], []
                for pred, gt in zip(pred_frames, gt_frames):
                    gt = _resize_like(gt, pred)
                    frame_psnr.append(float(compute_psnr(gt, pred)))
                    frame_ssim.append(float(compute_ssim(gt, pred)))
                per_video.append((float(np.mean(frame_psnr)), float(np.mean(frame_ssim))))

            psnr = float(np.mean([item[0] for item in per_video]))
            ssim = float(np.mean([item[1] for item in per_video]))
            pl_module.log("val/psnr_mean", psnr, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)
            pl_module.log("val/ssim_mean", ssim, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)
            mainlogger.info(
                "[video_inpaint_val] step=%d videos=%d psnr=%.4f ssim=%.4f "
                "target=winner_gt generated=videodpo_sample metrics=%s",
                step,
                count,
                psnr,
                ssim,
                metrics_path,
            )
        except Exception as exc:
            mainlogger.warning(f"[video_inpaint_val] step={step} failed: {type(exc).__name__}: {exc}")
        finally:
            if was_training:
                pl_module.train()
            torch.cuda.empty_cache()
