# Exp16 Stage1-500 Visual Sanity Context Audit

Date: 2026-06-17

## Status

`current_status = visual_sanity_completed`

This audit is for the small Exp16 sanity gate only. It does not authorize full
prior caching, Stage1 2000, or Stage2 training.

## Required Artifacts

| Item | Status | Path |
|---|---|---|
| Exp16 Stage1-500 checkpoint | exists | `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260617_exp16_limit100_exp16_prior_confidence_s1_500_limit100_pai/last_weights` |
| Exp16 eval hybrid checkpoint | exists | `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/hybrid/20260617_exp16_stage1_500_limit100_dpoS1_sftS2/last_weights` |
| Exp16 dpo_diag | exists | `exp16_prior_confidence_gated_dpo/dpo_diag/stage1_500_dpo_diagnostics.csv` |
| ProPainter prior cache limit100 | exists | `/mnt/nas/hj/H20_Video_inpainting_DPO/data/cache/exp16_propainter_prior_cache_limit100` |
| Exp16 prior manifest limit100 | exists | `/mnt/nas/hj/H20_Video_inpainting_DPO/data/cache/exp16_propainter_prior_cache_limit100/manifests/exp16_train_with_prior_limit100.jsonl` |
| DAVIS eval path | exists | `/mnt/workspace/hj/nas_hj/data/external/davis_432_240` |

## Eval Readiness

Exp16 can be evaluated as a Stage1-only intervention only by using a DPO-S1 +
SFT-S2 hybrid checkpoint. Directly loading the Stage1-only `last_weights` into
the DAVIS evaluator fails because it does not carry the Stage2 motion config.
The eval sanity run therefore used:

```text
Exp16_stage1_500_limit100 = Stage1-500 DPO + SFT-48000 Stage2 hybrid
```

This is the same stage-composition convention used for Stage1-only DPO rows:
`DPO-S1 + SFT-S2`.

## Comparison Paths

| Method | Path |
|---|---|
| SFT-48000 DiffuEraser | `/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000` |
| Exp11 outer b0.75 S2 | `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s2_2000_davis_pai/last_weights` |
| Exp16 Stage1-500 eval hybrid | `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/hybrid/20260617_exp16_stage1_500_limit100_dpoS1_sftS2/last_weights` |

## DAVIS10 Output

PAI output:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp16_stage1_500_visual_sanity_davis10
```

HAL synced output:

```text
/home/hj/dpo-2-1-exp/exp16_stage1_500_visual_sanity_davis10
```

Videos:

```text
boat, rhino, dog-agility, blackswan, lucia, bear, dance-jump, soccerball, kite-surf, breakdance
```

Protocol:

```text
raw6, no PCM, no mask dilation, no Gaussian blur, hard comp, frame-wise metric, no VBench
```
