# Exp57 Adaptive Transition-Safe Loss Implementation

Status: `EXP57_ADAPTIVE_TRANSITION_LOSS_READY`

Implemented isolated VOID-only loss mode:

`void_adaptive_transition_safe_dpo_v0`

Files:

- `exp57_void_adaptive_transition_safe/adaptive_transition_loss.py`
- `exp57_void_adaptive_transition_safe/tests/test_adaptive_transition_loss.py`

The implementation is isolated from VOID official source, `inference/metrics.py`, the shared trainer, DiffuEraser, and VideoPainter.

## Mechanisms

1. Adaptive loser safe-lambda:
   - Computes gradient dot/cosine.
   - Sets loser lambda to zero when loser gradient conflicts with winner gradient.
   - Clips lambda to `[0, lambda_max]`.

2. Adaptive transition safety:
   - Tracks object, overlap, affected, boundary, and outside deltas.
   - Downscales object DPO when transition risk is detected.
   - Increases overlap / affected / boundary preservation weights when risk rises.

3. Backtracking / finite-difference safety:
   - Evaluates candidate update scales `[1.0, 0.5, 0.25, 0.125, 0.0625]`.
   - Accepts the first scale that does not worsen winner, outside, overlap, affected, or boundary beyond configured thresholds.
   - Rejects the update if no safe scale exists.

4. Difficulty-aware configuration:
   - H20 cells have `ATS0`, `ATS_STRICT`, `ATS_HALFLR`, and `ATS_NODPO`.
   - PAI cells can reuse `ATS_SDPO`/`ATS_LINEAR`-style settings through the same core primitives.

## Boundaries

- No VOR-Eval.
- No hard comp.
- No 10-step.
- No third-backbone evidence claim.
