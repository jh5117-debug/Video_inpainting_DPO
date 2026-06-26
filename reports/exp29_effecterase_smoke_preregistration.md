# Exp29 EffectErase Smoke Pre-Registration

Date: 2026-06-26

Status: `EFFECTERASE_SMOKE_PREREGISTERED`

This locks EffectErase inference-smoke inputs and protocol before any
EffectErase output is generated or reviewed. It does not run inference, choose
parameters from outputs, train an adapter, or create a scientific-positive
claim.

EffectErase remains an OR baseline / diagnostic candidate. Because EffectErase
is trained on VOR, VOR diagnostic results cannot be used as primary on-policy
loser evidence or held-out scientific proof.

## Assets

- EffectErase repository:
  `/home/hj/video_inpainting_third_party/EffectErase`
- EffectErase repo commit: `bcee0a5da5ef387c2ba39390dc4d579503669fb8`
- License: `CC BY-NC 4.0`
- Recovered asset root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_weight_recovery_20260626`
- LoRA checkpoint:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_weight_recovery_20260626/FudanCVL/EffectErase/EffectErase.ckpt`
- Wan base:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_weight_recovery_20260626/Wan-AI/Wan2.1-Fun-1.3B-InP`

The previous weight-recovery milestone verified 19 asset manifest entries and
all returned `OK`.

## Official Command Anchor

The official `script/test_remove.sh` calls:

```bash
python examples/remove_wan/infer_remove_wan.py \
  --fg_bg_path <mp4> \
  --mask_path <mp4> \
  --output_path <mp4> \
  --text_encoder_path <Wan models_t5...pth> \
  --vae_path <Wan Wan2.1_VAE.pth> \
  --dit_path <Wan diffusion_pytorch_model.safetensors> \
  --image_encoder_path <Wan models_clip...pth> \
  --pretrained_lora_path <EffectErase.ckpt>
```

`infer_remove_wan.py` defaults to 81 frames. The currently verified reusable
Exp29 diagnostic sources are 17-frame materializations from the MiniMax
source32 gate. Therefore this smoke explicitly uses `--num_frames 17` and is
classified as diagnostic compatibility smoke rather than an official full
benchmark.

## Locked Inputs

- Manifest:
  `exp29_or_adapter_feasibility/manifests/effecterase_smoke_preregistered.jsonl`
- Rows: 6
- REAL / BLENDER balance: 3 / 3
- small / medium / large mask balance: 2 / 2 / 2

| sample_id | source_type | mask_bucket | scene_group |
| --- | --- | --- | --- |
| REAL_ENV249_00103_004_04 | REAL | small | REAL_ENV249_00103 |
| REAL_ENV231_00010_003_03 | REAL | medium | REAL_ENV231_00010 |
| REAL_ENV166_00002_001_02 | REAL | large | REAL_ENV166_00002 |
| BLENDER_FOREST026_00020 | BLENDER | small | BLENDER_FOREST026 |
| BLENDER_BEDROOM009_00083 | BLENDER | medium | BLENDER_BEDROOM009 |
| BLENDER_FOREST010_00004 | BLENDER | large | BLENDER_FOREST010 |

Every row points to verified 17-frame condition, winner, and mask frame
directories under the Exp29 MiniMax source32 materialization root.

## Data-Risk Labels

All rows are locked with:

- `source_role = vor_diagnostic_only`
- `vor_eval = false`
- `eligible_for_training = false`
- `scientific_role = diagnostic_only_vor_confounded`

No VOR-Eval row is used because no verified local VOR-Eval EffectErase input
pair was available at preregistration time. No non-VOR OR row is used because
no verified local non-VOR condition/mask/winner triplet was available.

## Fixed Inference Protocol

- Task: removal
- Condition: `fg_bg` video
- Mask: task mask video
- Winner/GT: `bg` video, diagnostics only
- Output: raw EffectErase output
- Diagnostic comp: optional post-hoc visualization only
- Frames: 17
- Resolution: 832 x 480
- Seed: 2025
- CFG: 1.0
- Steps: 50
- Frame interval: 1
- Hard comp: not used as OR output

## Promotion Rules

This smoke can only support:

- `EFFECTERASE_OR_BASELINE_READY`, if inference runs, videos decode, metrics are
  computed, and per-video review finds usable baseline behavior.
- `EFFECTERASE_INFERENCE_SMOKE_FAILED`, if model execution or video quality
  fails.
- `EFFECTERASE_BASELINE_BLOCKED`, if assets or runtime remain unusable.

It cannot support:

- `EFFECTERASE_TRUE_ADAPTER_FEASIBILITY_CONFIRMED`
- `SCIENTIFIC_POSITIVE`
- `UNIVERSAL_ADAPTER`

Adapter feasibility requires a later training-forward audit, zero-gap, and
one-step strict reload under non-leaky data semantics.
