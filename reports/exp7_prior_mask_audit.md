# Exp7 Prior / Mask Audit

Updated: 2026-06-04

## Scope

This audit reviews the current Exp7 partial-mask task before any further D3 / YouTube-VOS DPO expansion.
No training, data generation, checkpoint editing, or old-output deletion was performed.

## Findings

### 1. Exp7 training task

Current Exp7 launchers set the task-alignment flags correctly:

```text
TRAIN_MASK_MODE=partial
MASK_FROM_MANIFEST=true
LOSS_REGION_MODE=full
BETA_DPO=10
LOSE_GAP_WEIGHT=0.25
WINNER_ABS_REG_WEIGHT=0.05
WINNER_GAP_REG_WEIGHT=1.0
```

So the training intent is genuinely partial-mask DPO, not the old full-mask bridge.

### 2. Exp7 partial-mask eval path appears to omit ProPainter prior

`tools/eval_generated_loser_partialmask_model.py` loads DiffuEraser through
`tools/generate_diffueraser_fullmask_vbench.py::build_pipeline`, then calls the
pipeline with:

```text
images = masked_winner_images(winner_frames, mask_frames)
pipeline(..., images=images, masks=mask_frames, ...)
```

This eval path does not call `inference/run_OR.py`, does not pass
`propainter_model_dir`, and does not build a ProPainter/flow prior.

By contrast, `DPO_finetune/infer_diffueraser_candidate.py` uses the original OR
inference path and requires:

```text
--propainter_model_dir
--pcm_weights_path
inference/run_OR.py
```

Therefore current Exp7 partial-mask eval is likely a no-prior direct-pipeline
eval, not the normal DiffuEraser partial-mask OR prior path.

### 3. Why base can also look bad

If partial-mask inpainting eval omits ProPainter prior, even SFT/base
DiffuEraser may look poor on difficult masks. This means Exp7 bad videos cannot
be interpreted only as DPO failure until the correct prior path is tested.

### 4. Mask-size risk

The current canonical VideoDPO partial-mask policy default is 20%-30%:

```text
configs/generation/videodpo_partialmask_policy_v1_medium_hard_k4.yaml
mask_area_min ~= 0.20
mask_area_max ~= 0.30
```

User review suggests the masks are too large for the Exp7 debugging stage. The
new policy is 15%-20%:

```text
configs/generation/videodpo_partialmask_policy_v2_smallmask15_20_k4.yaml
mask_area_min = 0.15
mask_area_max = 0.20
```

### 5. Prior policy

| setting | prior policy |
| --- | --- |
| fullmask / video generation | no ProPainter prior; no context exists, use no-prior/noise prior |
| partial-mask video inpainting | ProPainter prior should be enabled; mask outside context is meaningful |

VideoDPO clips are often static, so prior benefits may be weak there. DAVIS /
YouTube-VOS have stronger motion, so prior may matter more.

### 6. SFT-48000 weight rule

All YouTube-VOS / D3 work must use the fine-tuned DiffuEraser SFT-48000 weights:

```text
PAI: /mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000
HAL local reference: /home/hj/Video_inpainting_DPO/weights/diffuEraser/converted_weights_step48000
```

H20 path could not be rechecked in this turn because SSH returned connection
reset / timeout. Do not guess; recheck before H20 YouTube-VOS/D3 work.

## Current Conclusion

Exp7 is suspicious, not simply failed. The likely audit issues are:

- partial-mask eval likely used a no-prior direct pipeline;
- mask size may be too large;
- base/SFT poor videos may be caused by the eval path or prior omission;
- DPO Stage2 still appears harmful and should stay disabled.

## Required Fix Before More D3 DPO

Create a new non-overwriting Exp7-fix data asset:

```text
data/generated_losers/exp07_fix_videodpo_smallmask15_20_prior_k4/
```

with:

```text
mask_area_min = 0.15
mask_area_max = 0.20
K = 4
prior_mode = propainter
selected_primary_comp.repaired.jsonl
selected_primary_nocomp.repaired.jsonl
```

Then run only Stage1 gates:

- PAI: `exp07_fix_smallmask_prior_wingap_lose025_stage1_gate1500`
- H20: `exp07_fix_smallmask_prior_wingap_nolose_stage1_gate1000`

Both must output four-column qualitative videos and dpo diagnostics.
