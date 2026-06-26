# Exp29 EffectErase Repo And Weight Audit

Date: 2026-06-26

## Status

- `EFFECTERASE_REPO_READY`
- `EFFECTERASE_BLOCKED_NO_WEIGHTS`
- `EFFECTERASE_INFERENCE_SMOKE_BLOCKED`
- `EFFECTERASE_TRAINABLE_FORWARD_NOT_VALIDATED`

EffectErase is not ready for Exp29 inference smoke because the official LoRA
checkpoint and Wan2.1-Fun InP model assets were not found in the audited local
or NAS paths.

## Repository

- Local repo: `/home/hj/video_inpainting_third_party/EffectErase`
- Remote: `https://github.com/FudanCVL/EffectErase.git`
- Commit: `bcee0a5da5ef387c2ba39390dc4d579503669fb8`
- License: CC BY-NC 4.0

## Official Asset Instructions

The README requires:

- `hf download alibaba-pai/Wan2.1-Fun-1.3B-InP --local-dir Wan-AI/Wan2.1-Fun-1.3B-InP`
- `hf download FudanCVL/EffectErase EffectErase.ckpt --local-dir ./`

The official removal script expects:

- `Wan-AI/Wan2.1-Fun-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth`
- `Wan-AI/Wan2.1-Fun-1.3B-InP/Wan2.1_VAE.pth`
- `Wan-AI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors`
- `Wan-AI/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`
- `EffectErase.ckpt`

Targeted checks did not find these assets under the expected local/NAS weight
paths.

## Code Evidence

Inference entry:

`examples/remove_wan/infer_remove_wan.py`

The script loads the Wan components and `EffectErase.ckpt` as a LoRA via
`model_manager.load_lora_v2(...)`, then calls the Wan removal pipeline with:

- `task="remove"`
- aligned `fg_bg` video
- aligned mask video
- removal prompt

Training utilities exist:

`examples/wanvideo/model_training/train.py`

The generic Wan training module calls `WanVideoPipeline.training_loss(...)`.
The local scheduler implements flow matching:

- `add_noise(sample, noise, timestep) = (1 - sigma) * sample + sigma * noise`
- `training_target(sample, noise, timestep) = noise - sample`

This makes adapter work technically plausible, but no removal-specific
policy/reference DPO gate has been run.

## Data Risk

The previous Exp26 compatibility audit records EffectErase as VOR-trained and
therefore not valid as primary on-policy VOR loser evidence. Until a non-VOR or
otherwise non-confounded validation design is locked, EffectErase should be used
only as:

- OR strong baseline;
- diagnostic;
- possible loser-generator ablation with explicit `primary_on_policy=false`.

It must not be used as GT or as primary VOR scientific proof.

## Blocker

Minimum user/actionable asset command, if the assets are permitted:

```bash
cd /home/hj/video_inpainting_third_party/EffectErase
hf download alibaba-pai/Wan2.1-Fun-1.3B-InP --local-dir Wan-AI/Wan2.1-Fun-1.3B-InP
hf download FudanCVL/EffectErase EffectErase.ckpt --local-dir ./
```

No fallback model was used.

