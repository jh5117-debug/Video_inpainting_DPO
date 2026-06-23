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
