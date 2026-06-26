# Exp29 Status

Current status: `READBACK_AND_SCAFFOLD_CREATED`

Exp29 is an isolated feasibility audit for MiniMax-Remover and EffectErase.
It inherits the Exp26 conclusion that DiffuEraser plus VideoPainter support
`CROSS_BACKBONE_ADAPTER_EVIDENCE_CONFIRMED`, but not a universal-adapter
claim.

No GPU inference, trainable-forward gate, DPO step, or long training has been
launched by the scaffold milestone.

## 2026-06-26 Repo And Weight Audit

- MiniMax: `MINIMAX_REPO_READY`, `MINIMAX_WEIGHTS_READY`.
- EffectErase: `EFFECTERASE_REPO_READY`, `EFFECTERASE_BLOCKED_NO_WEIGHTS`.
- No inference smoke or trainable-forward gate has run yet.

Reports:

- `reports/exp29_minimax_repo_weight_audit.md`
- `reports/exp29_minimax_repo_weight_audit.csv`
- `reports/exp29_minimax_asset_matrix.json`
- `reports/exp29_effecterase_repo_weight_audit.md`
- `reports/exp29_effecterase_repo_weight_audit.csv`
- `reports/exp29_effecterase_asset_matrix.json`

## 2026-06-26 Inference Smoke And Trainable Forward

- MiniMax inference: `MINIMAX_INFERENCE_SMOKE_PASSED_WITH_VISUAL_QUALITY_RISKS`.
- MiniMax visual quality: mixed; one medium-hard candidate and three
  trivial-bad outputs.
- MiniMax trainable forward: `MINIMAX_TRAINABLE_FORWARD_PASSED`.
- EffectErase inference: `EFFECTERASE_INFERENCE_SMOKE_BLOCKED_NO_WEIGHTS`.
