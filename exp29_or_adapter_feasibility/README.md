# Exp29 OR Adapter Feasibility

Exp29 audits MiniMax-Remover and EffectErase as object-removal baselines,
loser generators, and possible future true DPO adapter backbones.

This track starts from the Exp26 VideoPainter post-confirmation state and does
not modify Exp26 results, left-side CLI worktrees, shared trainers, or
`inference/metrics.py`.

Hard stops:

- No long training.
- No VideoPainter 100-step continuation.
- No RC-FPO.
- No VOR-Eval training, thresholding, or checkpoint selection.
- No universal-adapter or final-SOTA claim.

