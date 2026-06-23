# Exp25 Overnight Smoke6 Monitor

Date: 2026-06-23

## Runtime

- PAI controller PID: `1903925`
- Runtime root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623`
- Smoke6 output root: `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/preference_candidates/smoke6_canonical_raw6_d0`
- Visual review root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623/exp25_smoke6_visual_review`

## Technical Result

- Model: DiffuEraser
- Generator id: `diffueraser_or_none_propainter_abd3ad48f60f`
- PCM mode: `none`
- Prior mode: `propainter`
- Mask dilation: `0`
- Hard comp: `false`
- Samples completed: `6/6`
- Frames per sample: `24/24`
- Traceback/OOM/error grep: none observed in smoke log and per-sample logs.

## Semantic Check

Materialized manifest confirms:

- condition is `V_obj` / `FG_BG`
- winner is `V_bg` / `BG`
- mask is foreground object mask
- hard_comp is false

## Visual Review

| sample_id | judgement | note |
|---|---|---|
| REAL_ENV114_00004_004_02 | too_easy | Person removed cleanly; output very close to winner. |
| BLENDER_FOREST039_00117 | too_easy | Output nearly matches winner; weak preference signal. |
| BLENDER_FOREST039_00530 | too_easy | Output nearly matches winner; weak preference signal. |
| REAL_ENV024_00002_008_01 | slight_signal | Person removed; mild smoothing/fence/background differences. |
| BLENDER_CON001_00332 | too_easy | Dark scene output close to winner, no strong artifact. |
| REAL_ENV159_00010_003_05 | useful_hard | Clear body/hand/bag ghosting remains; usable hard negative, but may be severe. |

## Decision

Status: `TECHNICAL_PASS_QUALITY_YIELD_WEAK`.

The canonical d0 Smoke6 proves the current no-PCM DiffuEraser OR path can run
without hard comp or GT condition leakage, but the visual yield is not strong
enough to automatically promote Gate32 as a high-confidence data-generation
step. Gate32 can be run as yield calibration when a GPU is available, but the
controller must not treat Smoke6 completion alone as OR-DPO readiness.

EffectErase PAI inventory verification completed independently with `ok=true`,
37 files, no partials, no bad files, and contiguous VOR-Train parts `000-031`.
