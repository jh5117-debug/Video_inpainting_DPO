# Exp25 Status

- FULL_VOR_TRAIN_MASK_MEMBER_INDEX_COMPLETED
- TRIPLET_PAIRING_RULE_CONFIRMED
- AUDIT64_EXTRACTION_COMPLETED_WITH_ONE_BAD_SAMPLE_EXCLUDED
- GROUP_LEVEL_SPLITS_LOCKED
- GATE128_SELECTIVE_EXTRACTION_COMPLETED
- OR_GENERATION_NOT_STARTED

Latest commit on PAI at status write: $(git rev-parse HEAD)
## 2026-06-23 Gate128 Smoke

- Gate128 exact extraction and 6-sample materialization succeeded.
- ProPainter smoke passed: 6/6 samples, 24 raw frames each, no hard comp.
- DiffuEraser smoke blocked: PCM LoRA load path fails with `UNetMotionModel.load_lora_adapter` missing in the current environment.
- EffectErase smoke blocked: no verified inference wrapper/checkpoint path in Exp25.
- Full Gate128 OR loser generation has not been launched.
