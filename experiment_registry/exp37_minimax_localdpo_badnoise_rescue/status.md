# Exp37 Status

Current status: `MINIMAX_BAD_NOISE_STATES_READY`

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

## 2026-06-28 LocalDPO-style OR Corruption Pool

Current status: `LOCALDPO_STYLE_POOL_READY_VISUAL_REVIEW_PASS`

- Built train32/heldout16 local-corruption manifests from VOR-Train style
  Gate64 rows only; VOR-Eval used = `false`.
- Candidate rows per source <= `2`.
- Automatic selected usable: `39/48`.
- Codex final visual usable: `48/48`.
- Train32 final: `24` medium-hard, `8` hard-but-plausible.
- Heldout16 final: `14` medium-hard, `2` hard-but-plausible.
- No global collapse or systematic far-outside damage found in 48 selected
  review sheets.
- Next allowed milestone: diagnostic bad-noise scan. Training is still locked
  until bad-noise states and recipe preregistration are complete.

Reports:

- `reports/exp37_localdpo_style_or_corruption_pool.md`
- `reports/exp37_localdpo_style_or_corruption_pool.csv`
- `reports/exp37_localdpo_style_visual_review.csv`
- `reports/exp37_localdpo_style_or_corruption_pool_summary.json`

## 2026-06-28 Bad-Noise Diagnostic Scan

Current status: `MINIMAX_BAD_NOISE_STATES_READY`

- Scanned locked LocalDPO-style train32 rows only.
- Training launched: `false`.
- Model update: `false`.
- Candidate states per row: `64` (`K_noise=8`, `K_timestep=8`).
- Total candidate states: `2048`.
- Timesteps: `0.05, 0.15, 0.25, 0.35, 0.5, 0.65, 0.8, 0.95`.
- State manifest SHA256:
  `492210b2cd725faa348adcbafaf37bf82cc6790b4eb0607b9f758047d1c795d4`.
- Mean hard-A/random gradient proxy ratio: `0.570900`.
- Mean hard-A/random loser-local ratio: `0.331205`.

Interpretation: the outside-sane hard states are not simply the maximum raw
residual states. They are lower-gradient than the random baseline on average
because the random state often includes outside damage. Objective rescue must
therefore use the preregistered local hard states and outside preservation
terms rather than a blind residual maximization rule.

Reports:

- `reports/exp37_minimax_badnoise_diagnostic_scan.md`
- `reports/exp37_minimax_badnoise_diagnostic_scan.csv`
- `reports/exp37_minimax_badnoise_summary.json`
- `exp37_minimax_localdpo_badnoise_rescue/manifests/exp37_badnoise_states.jsonl`
