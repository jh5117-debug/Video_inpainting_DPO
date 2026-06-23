# VideoPainter v2 Adapter Ladder

Exp26 follows the project-wide backbone ladder:

1. L0 official baseline strict-load and native inference.
2. L1 native loss parity with official optimizer/scheduler/noising target.
3. L2 policy=reference DPO zero-gap test.
4. L3 one optimizer step plus strict checkpoint reload.
5. L4 ten-step smoke on real pairs.
6. L5 micro gates at 50, 100, 250, and 500 steps.
7. L6 promote to 1000/1500/2000 only if the curve is still improving.
8. L7 final test only after the checkpoint is locked.

This ladder exists so VideoPainter v2 does not repeat Exp14's pattern of
running a default 2000-step gate before proving native target parity, data
semantics, checkpoint identity, and checkpoint-curve quality.
