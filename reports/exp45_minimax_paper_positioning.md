# Exp45 MiniMax Paper Positioning

Status: `MINIMAX_DATA_READY_FOR_H20_STAGE2_TRAINING`

Exp45 produced a curated, human-reviewed same-source MiniMax Stage2 handoff package:

- formal split: `64/24/24`
- views: pseudo-success distillation, GT distillation, same-source preference
- scene overlap: `0`
- H20 filelist status: `EXP45_H20_FILELIST_READY`
- H20 missing paths on PAI: `0`
- training run by PAI: `false`
- optimizer step by PAI: `false`

Allowed wording:

- MiniMax has a curated same-source success/failure data package ready for H20 Stage2 training.
- MiniMax remains plumbing-positive, with a curated data package ready for
  training, but quality improvement is still unproven.

Forbidden wording:

- MiniMax third-backbone positive.
- universal adapter.
- all models supported.
- final SOTA.

DiffuEraser and VideoPainter remain the main positive adapter evidence. MiniMax is not counted as a third adapter until H20 training on this package passes search/shadow quality gates with real video review.
