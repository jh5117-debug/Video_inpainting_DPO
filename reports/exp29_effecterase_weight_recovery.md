# Exp29 EffectErase Weight Recovery

Date: 2026-06-26

Status: `EFFECTERASE_WEIGHTS_READY`

EffectErase official assets were recovered after the previous repo/weight audit
had blocked on missing weights. The files were downloaded on HAL and rsynced to
a writable Exp29 NAS cache on PAI. No fallback model was used.

This milestone only recovers and verifies assets. It does not run EffectErase
inference smoke, trainable forward, zero-gap, one-step, or any adapter training.

## Repository

- Local repo: `/home/hj/video_inpainting_third_party/EffectErase`
- Remote: `https://github.com/FudanCVL/EffectErase.git`
- Commit: `bcee0a5da5ef387c2ba39390dc4d579503669fb8`
- License: CC BY-NC 4.0

## Asset Root

PAI/NAS cache:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_weight_recovery_20260626`

The cache is under the Exp29 autoresearch output root because canonical shared
weights directories were not writable by `hj` during this milestone.

## SHA256 Verification

- Manifest entries checked: 19
- Result: all checked files `OK`
- Total cache size: 20G
- File count: 53

| File | SHA256 |
| --- | --- |
| `FudanCVL/EffectErase/EffectErase.ckpt` | `4e9ace4607a348a0d820328f827f7f4f751a0e44d4cf60665c03e9740dfa024d` |
| `Wan-AI/Wan2.1-Fun-1.3B-InP/Wan2.1_VAE.pth` | `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981` |
| `Wan-AI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors` | `8495d2b1673ffb18abb548a64ff3b0e4bd367734f653096f7a8a3ad46954d511` |
| `Wan-AI/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` | `628c9998b613391f193eb67ff68da9667d75f492911e4eb3decf23460a158c38` |
| `Wan-AI/Wan2.1-Fun-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth` | `7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d` |

## Data-Risk Reminder

EffectErase is VOR-trained according to the current paper/code audit. Even after
weight recovery, VOR results may only be used as strong baseline / diagnostic
evidence unless a non-confounded external validation design is locked.

Allowed next step:

- `EFFECTERASE_INFERENCE_SMOKE` for OR baseline/diagnostic quality.

Not allowed from this milestone alone:

- `EFFECTERASE_TRUE_ADAPTER_FEASIBILITY_CONFIRMED`
- scientific positive claims on VOR train/eval
- universal-adapter language
