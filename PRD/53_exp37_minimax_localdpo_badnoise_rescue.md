# Exp37 MiniMax LocalDPO-BadNoise Hybrid Rescue

Status: `EXP37_READBACK_COMPLETED`

Scope: MiniMax-only local corruption and bad-noise objective rescue after Exp36.
This lane must not repeat failed Exp36 recipes, must not run 30-step unless a
10-step rescue passes, must not start 2000-step, and must not claim universal
adapter.

Protected lanes: Exp31 VideoPainter 2000-step, Exp33 EffectErase VOR-Eval, and
left CLI remain read-only protected.

## 2026-06-28 Readback and Failure Taxonomy

Status: `EXP37_READBACK_COMPLETED`.

Exp37 was created from Exp36 HEAD
`3cd87e4b1a5b30a369ac3604086b7e31a4f45163`. Readback confirmed that MiniMax
is not failing because of checkpoint loading, ignored trained weights, or total
inability to learn. Exp36 sensitivity and winner-SFT controls show that
MiniMax changes outputs and lowers train loss, but the heldout quality remains
non-positive.

Prior visual evidence:

- Exp30 Gate64 adapter: visual better `0/32`.
- Exp35 rescue recipes: visual better `0/48`.
- Exp36 winner-SFT: visual better `0/24`.

Current failure taxonomy: data/objective/update-scale/generalization. Exp37
therefore must first diagnose train-vs-heldout behavior, then build a cleaner
LocalDPO-style corruption pool and only then scan bad-noise states. No 30-step
or long training is unlocked by readback.

Report:

- `reports/exp37_minimax_readback.md`
