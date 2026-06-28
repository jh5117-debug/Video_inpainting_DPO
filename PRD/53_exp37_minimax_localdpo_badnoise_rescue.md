# Exp37 MiniMax LocalDPO-BadNoise Hybrid Rescue

Status: `MINIMAX_BAD_NOISE_STATES_READY`

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

## 2026-06-28 Train-vs-Heldout Diagnosis

Status: `MINIMAX_OBJECTIVE_SIGNAL_TOO_WEAK`.

Evaluated the best Exp36 S1 winner-SFT/control checkpoint
`S1_adamw_lr1em05/checkpoint-10` on locked Gate64 `train16` and `heldout16`.
The run used PAI GPU0 only, wrote to the Exp37 NAS log root, and did not touch
Exp31, Exp33, or left CLI processes.

Metrics:

- Train16 mean delta full/mask/boundary/outside PSNR:
  `+0.029083` / `-0.008362` / `+0.001128` / `+0.048654`.
- Heldout16 mean delta full/mask/boundary/outside PSNR:
  `-0.010218` / `-0.008293` / `-0.010939` / `-0.014499`.
- Positive mask rows: train `9/16`, heldout `6/16`.
- Positive boundary rows: train `7/16`, heldout `5/16`.

Codex reviewed all `32/32` Step0-vs-Step10 temporal strips. Visual result:
`0` better, `32` tie/no visible change, `0` new artifacts. The train-side
videos do not show meaningful local OR-quality improvement, so the failure is
not a pure train-positive/heldout-negative generalization issue. The current
signal remains too weak at the objective/update level.

Allowed next step: build cleaner LocalDPO-style local corruption pairs and then
run the preregistered bad-noise diagnostic scan. No 30-step, 2000-step,
RC-FPO, or universal-adapter claim is unlocked.

Reports:

- `reports/exp37_minimax_train_vs_heldout_diagnosis.md`
- `reports/exp37_minimax_train_vs_heldout_metrics.csv`
- `reports/exp37_minimax_train_vs_heldout_visual_review.csv`
- `reports/exp37_minimax_train_vs_heldout_summary.json`

## 2026-06-28 LocalDPO-style Clean OR Corruption Pool

Status: `LOCALDPO_STYLE_POOL_READY_VISUAL_REVIEW_PASS`.

Built an Exp37-isolated LocalDPO-style OR corruption pool from the locked
Gate64 VOR-Train rows:

- condition = `V_obj`
- winner = `V_bg`
- loser = locally corrupted `V_bg`
- mask = foreground object mask
- affected map = `abs(V_obj - V_bg)` for L3/L4 local-region construction
- VOR-Eval used = `false`
- candidate rows per source <= `2`
- split remains scene-disjoint: train32 and heldout16 inherited from locked
  Gate64 train/heldout split.

Automatic selected usable was `39/48`, because outside PSNR counted
affected/boundary soft-region edits as outside damage in 9 rows. Codex opened
all `48/48` selected primary review sheets and wrote final classifications:

- Train32: `24` medium-hard, `8` hard-but-plausible.
- Heldout16: `14` medium-hard, `2` hard-but-plausible.
- Final usable: `48/48`.
- No global collapse, black/purple failure, or systematic far-outside damage
  was observed.

Manifest SHA256:

- Train32: `1ed5c7d4667e7ad1ddc26a042a8613a2a3135c8a8bd3da37071e39e608b66269`
- Heldout16: `c761ec3115bf28305879cfd5a9ea835eb121cb9f95ece1e1dcdce99e95ff4abf`

This unlocks the diagnostic bad-noise scan only. It does not unlock 10-step
training until bad-noise states and recipe preregistration are complete, and it
does not unlock 30-step or long training.

Reports and manifests:

- `reports/exp37_localdpo_style_or_corruption_pool.md`
- `reports/exp37_localdpo_style_or_corruption_pool.csv`
- `reports/exp37_localdpo_style_visual_review.csv`
- `reports/exp37_localdpo_style_or_corruption_pool_summary.json`
- `exp37_minimax_localdpo_badnoise_rescue/manifests/localdpo_or_train32.jsonl`
- `exp37_minimax_localdpo_badnoise_rescue/manifests/localdpo_or_heldout16.jsonl`

## 2026-06-28 MiniMax Bad-Noise Diagnostic Scan

Status: `MINIMAX_BAD_NOISE_STATES_READY`.

Ran the preregistered diagnostic scan on PAI GPU0 using the locked
LocalDPO-style train32 manifest. This was a no-training scan:

- Training launched: `false`.
- Model update: `false`.
- Train rows: `32`.
- Candidate states per row: `64` (`K_noise=8`, `K_timestep=8`).
- Total candidate states: `2048`.
- Timesteps: `0.05, 0.15, 0.25, 0.35, 0.5, 0.65, 0.8, 0.95`.
- State manifest SHA256:
  `492210b2cd725faa348adcbafaf37bf82cc6790b4eb0607b9f758047d1c795d4`.

The scan selected `hard_state_A`, `hard_state_B`, and `hard_state_C` for each
train row. `hard_state_A` applies an outside-sanity filter before maximizing
local loser residual. As a result, it is intentionally not the globally largest
residual state: mean hard-A/random gradient proxy ratio was `0.570900`, and
mean hard-A/random loser-local ratio was `0.331205`. This is useful diagnostic
evidence that the rescue recipes must use locally controlled hard states rather
than simply maximizing raw residuals, which often includes outside damage.

Observed selected-state tendencies:

- `hard_state_A`: mostly `t=0.25` (`28/32` rows), mean gradient proxy
  `1.449670`, mean loser-local score `1.119573`.
- `hard_state_B`: split between `t=0.25` and `t=0.95`, mean gradient proxy
  `1.204644`, mean loser-local score `0.728243`.
- `hard_state_C`: mostly `t=0.25` (`30/32` rows), mean gradient proxy
  `1.430135`, mean loser-local score `1.070902`.

This unlocks only objective-recipe preregistration. It does not unlock 10-step
training until the recipes are explicitly preregistered, and it does not unlock
30-step or long training.

Reports and manifests:

- `reports/exp37_minimax_badnoise_diagnostic_scan.md`
- `reports/exp37_minimax_badnoise_diagnostic_scan.csv`
- `reports/exp37_minimax_badnoise_summary.json`
- `exp37_minimax_localdpo_badnoise_rescue/manifests/exp37_badnoise_states.jsonl`
