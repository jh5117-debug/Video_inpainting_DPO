# Exp37 Status

Current status: `EXP37_READBACK_COMPLETED`

## 2026-06-28 Readback

- Branch: `research/exp37-minimax-localdpo-badnoise-rescue-20260627`.
- Base: `origin/research/exp36-minimax-objective-rescue-20260627`.
- Start HEAD: `3cd87e4b1a5b30a369ac3604086b7e31a4f45163`.
- Exp36 ruled out checkpoint/load and ignored-weight failures.
- Exp36 did not produce heldout quality-positive MiniMax evidence.
- Prior visual better rows: Exp30 `0/32`, Exp35 `0/48`, Exp36 `0/24`.
- Next allowed milestone: train-vs-heldout diagnosis.
- 30-step, 2000-step, RC-FPO, and universal-adapter language remain forbidden.

Report:

- `reports/exp37_minimax_readback.md`

## 2026-06-28 Train-vs-Heldout Diagnosis

Current status: `MINIMAX_OBJECTIVE_SIGNAL_TOO_WEAK`

- Evaluated Exp36 S1 `checkpoint-10` on locked Gate64 `train16` and
  `heldout16`.
- Train16 mean mask/boundary PSNR deltas: `-0.008362` / `+0.001128`.
- Heldout16 mean mask/boundary PSNR deltas: `-0.008293` / `-0.010939`.
- Codex visual review: `32/32` strips reviewed, `0` better, `32` tie/no
  visible change, `0` new artifact.
- Diagnosis: train-side output is not meaningfully improved, so the MiniMax
  issue is objective/update signal too weak rather than pure generalization
  failure.
- Next allowed milestone: LocalDPO-style clean OR corruption subset.

Reports:

- `reports/exp37_minimax_train_vs_heldout_diagnosis.md`
- `reports/exp37_minimax_train_vs_heldout_metrics.csv`
- `reports/exp37_minimax_train_vs_heldout_visual_review.csv`
- `reports/exp37_minimax_train_vs_heldout_summary.json`
