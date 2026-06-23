# Exp25 Gate128 Three-Model OR Loser Smoke

Date: 2026-06-23  
Scope: 6 balanced samples from `vor_gate128.jsonl` materialized to exact 24-frame image/mask folders.

## OR Semantics

- condition: `FG_BG` / `V_obj`
- winner: `BG` / `V_bg`
- mask: foreground object mask
- loser target: raw model output
- hard comp: false
- comp mode: none
- VOR-Eval: not used

## Materialization

Command ran on PAI and produced:

`exp25_vor_or_preference_data/manifests/vor_gate128_smoke6_24f_materialized.jsonl`

Output root:

`/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/materialized/gate128_smoke6_24f`

Result: `ok=6`, `failed=0`.

## Model Smoke Results

| Model | Samples | OK | Failed | Status | Notes |
|---|---:|---:|---:|---|---|
| ProPainter | 6 | 6 | 0 | PASS | generated 24 raw frames per sample; no hard comp |
| DiffuEraser | 6 | 0 | 6 | BLOCKED | all failed at PCM LoRA loading compatibility |
| EffectErase | 6 | 0 | 0 | BLOCKED | no verified EffectErase inference wrapper/checkpoint path in Exp25 worktree |

## DiffuEraser Blocker

Representative failing log:

`/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp25_vor_or_preference_data/gate128_smoke/diffueraser/logs/REAL_ENV114_00004_004_02.log`

Error:

`AttributeError: 'UNetMotionModel' object has no attribute 'load_lora_adapter'`

The error occurs when the copied DiffuEraser OR path calls `pipeline.load_lora_weights(...)` for PCM weights. Because disabling PCM or silently skipping LoRA would change the generator identity, Gate128 DiffuEraser generation was not launched.

## ProPainter Output

Output root:

`/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp25_vor_or_preference_data/gate128_smoke/propainter/raw_frames`

All six outputs contain 24 raw frames. No hard-composited outputs were produced or used.

## Decision

Gate128 full loser generation is **not ready**.

Safe next actions:

1. Implement an Exp25-isolated DiffuEraser OR compatibility wrapper that preserves the intended generator identity and does not fallback.
2. Resolve or explicitly register a verified EffectErase inference wrapper/checkpoint path.
3. Re-run the 6-sample smoke before launching DiffuEraser Gate128, EffectErase Gate32, or ProPainter Gate32 production generation.
