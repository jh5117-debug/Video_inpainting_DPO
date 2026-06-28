# Exp37 MiniMax Readback

Branch: `research/exp37-minimax-localdpo-badnoise-rescue-20260627`

Base: `origin/research/exp36-minimax-objective-rescue-20260627`

Start HEAD: `3cd87e4b1a5b30a369ac3604086b7e31a4f45163`

## Files Read

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/50_exp30_vor_or_multimodel_minimax.md`
- `PRD/51_exp35_minimax_flow_dpo_rescue.md`
- `PRD/52_exp36_minimax_objective_rescue.md`
- `experiment_registry/exp30_vor_or_multimodel_minimax/status.md`
- `experiment_registry/exp35_minimax_flow_dpo_rescue/status.md`
- `experiment_registry/exp36_minimax_objective_rescue/status.md`
- `reports/exp30_minimax_gate64_adapter_10step_v3.md`
- `reports/exp30_minimax_gate64_adapter_10step_metrics_v3.csv`
- `reports/exp30_minimax_gate64_adapter_10step_visual_review_v3.csv`
- `reports/exp35_minimax_rescue_10step.md`
- `reports/exp35_minimax_rescue_10step_metrics.csv`
- `reports/exp35_minimax_rescue_10step_visual_review.csv`
- `reports/exp36_minimax_nochange_forensic_audit.md`
- `reports/exp36_minimax_inference_sensitivity.md`
- `reports/exp36_minimax_trainable_scope_audit.md`
- `reports/exp36_minimax_winner_sft_positive_control.md`
- `reports/exp36_minimax_winner_sft_visual_review.csv`
- `reports/exp36_minimax_paper_positioning.md`

## What Exp36 Ruled Out

Exp36 ruled out the easy failure modes:

- Not a checkpoint/load fallback: Step10 outputs and sensitivity tests showed nonzero response.
- Not inference ignoring trained weights: identity replay MAE was `0.0`, while a temporary `1.01x` transformer perturbation produced full/mask MAE `0.088218 / 0.156302`.
- Not total inability to learn: winner-SFT reduced train loss, moved parameters, strict-reloaded checkpoints, and changed outputs.
- Not collapse: prior Exp35 and Exp36 reviews found no black/purple collapse for the stable recipes.

## What Remains Unresolved

The unresolved issue is quality transfer from training signal to heldout repair:

- Exp30 frozen/EMA 10-step: visual better `0/32`.
- Exp35 R1/R2/R3 hard-noise rescue: visual better `0/48`.
- Exp36 winner-SFT positive-control: visual better `0/24`.
- Exp36 S1 LoRA had a tiny best mask PSNR delta `+0.000986`, with boundary delta `-0.004270`, and no visible improvement.

Current failure class: data/objective/update-scale/generalization.

## Train vs Heldout State Before Exp37

Exp36 proved that training loss decreases, but did not prove that train videos visually improve. Exp37 must explicitly compare train-side and heldout-side Step0 vs Step10 before attempting new DPO-style recipes.

If train improves but heldout does not, the status should become `MINIMAX_GENERALIZATION_FAILURE`.

If train does not improve, the status should become `MINIMAX_OBJECTIVE_SIGNAL_TOO_WEAK`.

If both improve, only then should Exp37 consider the rescue promising.

## Loser Quality Taxonomy

Exp30/Exp35 used a mixed OR pool with controlled corruption, MiniMax, ProPainter, and one DiffuEraser row. The pool was medium-hard enough to run the original gate, but the preference objective still produced ties/slight degradations. The remaining concern is that current losers may be too diverse, too noisy, or not locally aligned enough for MiniMax flow-DPO to learn stable affected-region repair.

## Why LocalDPO-Style Corruption Is Next

LocalDPO-style controlled corruption is the next safest data direction because it can:

- keep the clean winner as `V_bg`;
- restrict defects to object/affected/boundary regions;
- preserve outside content by construction;
- avoid uncontrolled MiniMax-generated trivial-bad artifacts;
- make the loser defect vector simpler and more local than the mixed OR pool.

This does not guarantee success. It only creates a cleaner diagnostic split for MiniMax.

## Safe Use of MiniMax Bad-Noise

Bad-noise should be used diagnostically and preregistered before training:

- scan fixed `K_noise=8`, `K_timestep=8` states;
- select hard states by local loser residual while checking outside sanity;
- compare hard-state vs random-state gradient scale;
- fix utility scale before any training;
- do not choose states by heldout video quality after the fact.

## Protected Lane Status

PAI readback found GPU0 free, GPU1 reserved by `cli4` locks, and GPU2-7 occupied by other `/usr/local/bin/python3.10` jobs. Exp37 has not sent any signal, reset any GPU, or modified Exp31, Exp33, or left-CLI files.

## Promotion Gate

Exp37 may proceed only in this order:

1. Train-vs-heldout diagnosis.
2. LocalDPO-style clean corruption pool.
3. Bad-noise diagnostic scan.
4. Preregister exactly three recipes.
5. Run 10-step rescue.
6. Run 30-step only if 10-step is positive.

No universal-adapter, third-backbone success, RC-FPO, 2000-step, or blind 30-step language is allowed from readback alone.

Status: `EXP37_READBACK_COMPLETED`.
