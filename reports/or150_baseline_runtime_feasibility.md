# OR150 Baseline Runtime Feasibility

Date: 2026-06-16

## Summary

The OR150 dataset is staged, but the full 8-method table must be gated by
runnable inference. Current status:

| Method | Status | Meaning |
|---|---|---|
| DiffuEraser SFT-48000 | READY | Project OR wrapper and SFT-48000 weights are present. |
| DiffuEraser Exp11 outer b0.75 S2 | READY_WITH_OR_RERUN | Best current method; BR numbers exist, OR true-mask numbers still need rerun. |
| MiniMax-Remover | READY_BUT_NEEDS_ISOLATED_ENV | Frozen baseline only; inference wrapper/repo/weights are available, but current DiffuEraser env is too old for MiniMax's diffusers dependency. |
| COCOCO | READY | Real weights and official repo path exist; run one-video env smoke before full run. |
| VideoPainter | READY | Frozen baseline can be evaluated with an Exp14-style thin adapter. |
| FloED | BLOCKED_MISSING_LOCAL_RUNTIME | No verified repo/weights/wrapper yet. |
| VACE | BLOCKED_MISSING_LOCAL_RUNTIME | No verified repo/weights/wrapper yet. |
| VideoComp / VideoComposer | BLOCKED_MISSING_OR_RUNTIME | No clean OR-compatible runtime validated. |

## Dataset

The OR benchmark uses:

```text
exp15_or_benchmark/manifests/or150_manifest.csv
```

with 50 DAVIS2017 foreground-mask videos and 100 YouTubeVOS videos.

## Weight Findings

### MiniMax-Remover

HAL downloaded a transient copy from:

```text
zibojia/minimax-remover
```

The files are real safetensors, not LFS pointers:

```text
vae/config.json
vae/diffusion_pytorch_model.safetensors
transformer/config.json
transformer/diffusion_pytorch_model.safetensors
scheduler/scheduler_config.json
```

PAI already has an equivalent real copy via the historical `current` symlink:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/weights/minimax_remover/current
-> /mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/minimax
```

The public MiniMax repo is inference-oriented in this project context. It is
valid as a frozen OR baseline, not as a trainable DPO adapter.

Official repo synced to PAI/NAS:

```text
/mnt/nas/hj/official_repos/MiniMax-Remover_28e12b4
```

### COCOCO

PAI/NAS contains real COCOCO weights:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/COCOCO_weight
```

It includes:

```text
cococo/model_0.pth
cococo/model_1.pth
cococo/model_2.pth
cococo/model_3.pth
stable-diffusion-v1-5-inpainting/
```

Official repo synced to PAI/NAS:

```text
/mnt/nas/hj/official_repos/COCOCO_9ebe984
```

Before full OR150, run a one-video environment smoke to catch Python/package
issues.

### VideoPainter

PAI/NAS contains VideoPainter and CogVideoX weights under:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter/ckpt
```

VideoPainter should be evaluated as a frozen baseline for OR, not using the
failed Exp14 DPO adapter.

## PAI Environment Check

PAI has only the shared DiffuEraser env available at:

```text
/mnt/nas/hj/conda_envs/diffueraser
```

Import-level checks:

| Method | Result |
|---|---|
| COCOCO | import-level OK in DiffuEraser env (`cococo`, `diffusers`, `omegaconf`, `einops`). |
| MiniMax-Remover | import fails in DiffuEraser env because `diffusers==0.29.2` lacks `AutoencoderKLWan` and `FP32LayerNorm`. |

MiniMax's public requirements request `diffusers==0.33.1` and a newer torch
stack. A first isolated venv attempt under `/mnt/nas/hj/conda_envs/` was
aborted because pip stalled on NAS. The partial venv was removed. Do not upgrade
the shared DiffuEraser env.

## Immediate Run Order

1. Run a 1-video OR smoke on PAI for `diffueraser_sft`, `exp11_outer_b075_s2`,
   `cococo`, and `videopainter` in the existing envs.
2. Build MiniMax in an isolated local-disk env or container, then run a 1-video
   smoke. Do not change the shared DiffuEraser env.
3. If smoke passes, launch OR150 inference for the ready methods.
4. Generate the 8-method visualization grid with blocked methods shown as
   unavailable only after runnable outputs exist for the requested methods.
5. Keep FloED/VACE/VideoComp blocked until their local OR runtime is verified.

## Do Not Do

- Do not use BR/DAVIS432 masks as DAVIS OR masks.
- Do not report the existing Exp11 BR numbers as true OR150 numbers.
- Do not fabricate FloED/VACE/VideoComp results.
- Do not commit weights, frames, mp4s, or generated outputs.
