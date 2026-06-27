# Exp36 MiniMax Trainable Scope Audit

Status: `MINIMAX_TRAINABLE_SCOPE_EXPANDED_S1_READY`

No training and no inference were launched in this milestone.

## Current Scope

Exp30/Exp35 used the current full MiniMax transformer scope:

- Status: `MINIMAX_TRAINABLE_SCOPE_CURRENT_OK`
- Checkpoint tensors: `461`
- Represented parameters: `1127055424`
- LoRA / adapter tensors in current checkpoint: `0`
- Inference sensitivity: `MINIMAX_INFERENCE_SENSITIVITY_PASS`

The current scope is not ignored by inference and is not too small, but full
transformer updates have so far moved outputs in non-useful directions.

## Prepared Scopes

S0:

- Current full MiniMax transformer scope.
- Available for winner-SFT positive-control only.
- High risk of overfitting / outside drift based on Exp35.

S1:

- LoRA on DiT self-attention q/k/v/out and output projection families.
- Rank `8`, alpha `16`, dropout `0`.
- Isolated Exp36 helper and tests added.
- Status: ready for a bounded positive-control after this audit.

S2:

- S1 plus last-four-block MLP LoRA.
- Rank `8`, alpha `16`, dropout `0`.
- Locked until S1 has weak positive-control evidence.

## Tests Added

- `test_minimax_trainable_scope_names.py`
- `test_minimax_lora_forward_usage.py`
- `test_minimax_checkpoint_roundtrip.py`
- `test_minimax_reference_frozen.py`

## Conclusion

Exp36 can continue to winner-SFT positive-control using S0 and optionally S1.
It still must not run 30-step or long training unless a later Exp36 10-step
objective rescue gate passes.

