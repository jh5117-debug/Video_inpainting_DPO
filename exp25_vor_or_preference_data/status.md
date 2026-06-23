# Exp25 Status

- FULL_VOR_TRAIN_MASK_MEMBER_INDEX_COMPLETED
- TRIPLET_PAIRING_RULE_CONFIRMED
- AUDIT64_EXTRACTION_COMPLETED_WITH_ONE_BAD_SAMPLE_EXCLUDED
- GROUP_LEVEL_SPLITS_LOCKED
- GATE128_SELECTIVE_EXTRACTION_COMPLETED
- OR_GENERATION_NOT_STARTED
- DIFFUSERASER_CANONICAL_RAW6_NO_PCM_LOCKED_PENDING_SMOKE6

Latest commit on PAI at status write: $(git rev-parse HEAD)
## 2026-06-23 Gate128 Smoke

- Gate128 exact extraction and 6-sample materialization succeeded.
- ProPainter smoke passed: 6/6 samples, 24 raw frames each, no hard comp.
- DiffuEraser smoke blocked: PCM LoRA load path fails with `UNetMotionModel.load_lora_adapter` missing in the current environment.
- EffectErase smoke blocked: no verified inference wrapper/checkpoint path in Exp25.
- Full Gate128 OR loser generation has not been launched.

## 2026-06-23 Canonical No-PCM Correction

- Existing DiffuEraser Smoke6 used `mask_dilation_iter=8`; it remains a
  technical OR-style diagnostic and is not the canonical raw6 pass.
- Canonical Exp25 no-PCM config is now
  `DE_CANONICAL_RAW6_NO_PCM_PROP_PRIOR` with `mask_dilation_iter=0`.
- Gate32/Gate128 remain blocked until fresh canonical Smoke6 is run and
  visually reviewed.

## 2026-06-23 Overnight Autonomous Controller

- Controller launched on PAI with PID `1903925`.
- Runtime root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623`.
- PAI snapshots were created with git archive, not full worktree rsync.
- Exp25 controller snapshot: `a4b031bc8326f600756c724af062db68f0d9f7b3`.
- Fixed six-sample Gate128 Smoke6 materialization completed at:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/materialized/gate128_smoke6_canonical_d0_24f/smoke6_materialized.jsonl`.
- Fresh canonical DiffuEraser Smoke6 d0 remains queued because GPU2/4/5/6
  are all occupied by other jobs; GPU7 remains excluded.
- EffectErase remote inventory completed on HAL:
  revision `fa09dc61128ca0418a4a13364d97a08018ea9cc7`,
  required files `37`, total bytes `363730944386`.
- EffectErase PAI checksum verification is running as HAL background PID
  `3577126`.
- Gate32/Gate128 OR generation remains blocked until canonical d0 Smoke6
  completes and is visually reviewed.

## 2026-06-23 Overnight Smoke6 Monitor Update

- Fresh canonical d0 DiffuEraser Smoke6 completed on PAI: `6/6`, `24` frames each.
- Generator id: `diffueraser_or_none_propainter_abd3ad48f60f`.
- Confirmed `pcm_mode=none`, `prior_mode=propainter`, `mask_dilation_iter=0`, `hard_comp=false`.
- Visual review completed on six contact sheets.
- Decision: `TECHNICAL_PASS_QUALITY_YIELD_WEAK`.
- Only `REAL_ENV159_00010_003_05` showed a clearly useful hard-negative artifact; most samples were too close to winner.
- Gate32 remains a yield-calibration step, not an automatic OR-DPO readiness promotion.
- Report: `reports/exp25_overnight_smoke6_monitor.md`.
