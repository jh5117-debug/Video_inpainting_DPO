# Exp35 MiniMax Flow-DPO Rescue

Status: `EXP35_READBACK_COMPLETED`

Exp35 is an isolated rescue/debug track for the MiniMax flow-DPO adapter path.
It starts from Exp30 Gate64 V3 data readiness and the failed MiniMax 10-step
quality gate. The goal is not to add steps blindly, but to determine whether
the no-change behavior comes from checkpoint loading, inference sensitivity,
trainable scope, objective scale, learning rate, or missing bad-noise/hard
timestep selection.

## Ground Rules

- Do not continue VideoPainter 100-step or Exp31 2000-step.
- Do not touch Exp33 EffectErase VOR-Eval baseline.
- Do not touch left CLI / cli4 worktrees, locks, heartbeats, or processes.
- Do not modify `inference/metrics.py`, shared trainers, or Exp1-Exp29
  history.
- Do not overwrite Exp30 outputs, checkpoints, or reports.
- Do not run 30-step before an explicitly positive 10-step rescue recipe.
- Do not run 500/1000/2000-step, RC-FPO, or universal-adapter claims.

## 2026-06-27 Readback

- Branch: `research/exp35-minimax-flow-dpo-rescue-20260627`.
- Base branch: `origin/research/exp30-vor-or-multimodel-minimax-adapter-20260627`.
- Start HEAD: `f69688fe4ff96c4d4f0dcd308eef69822fc1035b`.
- Exp30 candidate-generation status: `VOR_OR_GATE64_MULTIMODEL_POOL_READY`.
- Exp30 MiniMax status: `MINIMAX_ADAPTER_RECIPE_NOT_READY`.
- Gate64 train32 SHA256:
  `1eda205d2dc48714269f30eb390d959549387a778e6438267e6aba087ba14196`.
- Gate64 heldout16 SHA256:
  `84c231ded930d740bf299b27c2a6b1e95d7decdb3665051371c5df90ae9f2ade`.

Exp30 failed as a quality-positive MiniMax adapter recipe, not as a data or
basic plumbing gate. The data pool exists, zero-gap passed, one-step strict
reload passed, no NaN/Inf occurred, and Step10 inference did load the saved
checkpoint. However, output movement was too small to be useful:

- Frozen mean heldout deltas: mask PSNR `-0.001068`, boundary PSNR
  `-0.002821`, outside PSNR `-0.006340`.
- EMA mean heldout deltas: mask PSNR `-0.001851`, boundary PSNR `-0.003092`,
  outside PSNR `-0.006033`.
- Codex visual review in Exp30: Step10 better `0/32`, tie/no visible
  improvement `32/32`, new visible artifact `0/32`.

Initial readback of the Exp30 script shows that the previous gate trained the
full MiniMax transformer, not a LoRA-only adapter scope. The recipe used
AdamW, LR `5e-7`, beta `1.0`, linear utility near `0.5`, deterministic
single-state timestep/noise per step, and no bounded bad-noise / hard-timestep
miner. The Step10 delta probe was only about `2.7e-7`, consistent with an
update-scale/objective-state problem.

Protected-lane readback:

- Exp31 was observed on GPU1, PID `1215136`, and is reserved.
- cli4 locks reserve GPU1/GPU2/GPU3/GPU4.
- Exp35 must only use eligible non-reserved GPUs and must not send signals to
  protected lanes.

Report:

- `reports/exp35_minimax_rescue_readback.md`

No GPU training, inference, 30-step, RC-FPO, or protected-lane action was
launched by this readback milestone.

## 2026-06-27 No-Change Forensic Audit

- Status: `MINIMAX_NOCHANGE_CAUSE_UTILITY_SCALE_TOO_WEAK`.
- Training performed in this milestone: false.
- Exp30 checkpoints audited: frozen and EMA `checkpoint-0` vs `checkpoint-10`.
- Common safetensor keys: 461 per recipe.
- Missing/unexpected keys: 0/0 for both recipes.
- Parameter count read: 1,127,055,424 per recipe.
- Frozen mean abs parameter delta: `1.5329060227168864e-08`.
- Frozen max abs parameter delta: `8.106231689453125e-06`.
- Frozen delta / param norm ratio: `5.6404525516172905e-06`.
- EMA mean abs parameter delta: `1.5302821461092914e-08`.
- EMA max abs parameter delta: `8.106231689453125e-06`.
- EMA delta / param norm ratio: `5.630459939756668e-06`.
- Step0 vs Step10 output rows compared: 32.
- Byte-identical rows: 0.
- Mean full / mask / affected / outside absolute pixel diff:
  `0.13143352206508793`, `0.18672874342540607`,
  `0.1731182035360047`, `0.10850902535158265`.
- Max absolute pixel diff: 28.

Loss / utility scale:

- Frozen linear utility mean/min/max:
  `0.4999982982873917` / `0.49997058510780334` /
  `0.5000085830688477`.
- Frozen abs margin mean: `2.8578052297234536e-05`.
- EMA linear utility mean/min/max:
  `0.5000003516674042` / `0.49999284744262695` /
  `0.5000050663948059`.
- EMA abs margin mean: `1.2780050747096539e-05`.
- t range in Exp30 diagnostics: `0.24` to `0.69`, mean `0.465`.

Conclusion:

The previous Exp30 Step10 checkpoint was not a silent fallback to Step0:
checkpoint keys match, parameter deltas are nonzero, and Step0/Step10 outputs
are not byte-identical. The movement is simply far below useful visual scale.
The dominant audited cause is a near-constant Linear-DPO utility around 0.5
with tiny margins, compounded by very small parameter movement. Exp35 should
next run an inference-sensitivity positive-control before redesigning recipes.

Reports:

- `reports/exp35_minimax_10step_forensic_audit.md`
- `reports/exp35_minimax_10step_forensic_audit.csv`
- `reports/exp35_minimax_10step_output_diff.csv`
- `reports/exp35_minimax_10step_param_delta.csv`
- `reports/exp35_minimax_10step_loss_scale.csv`
- `reports/exp35_minimax_10step_forensic_summary.json`

## 2026-06-27 Inference Sensitivity Positive-Control

- Status: `MINIMAX_INFERENCE_SENSITIVITY_PASS`.
- Training performed in this milestone: false.
- Rows: 4 total, using 2 heldout rows and 2 train rows from the locked Exp30
  Gate64 V3 manifests.
- GPU used: PAI GPU6 only; protected Exp31/cli4/other tasks were read-only
  audited and were not signaled.
- Step0 checkpoint:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp30_vor_or_multimodel_minimax/minimax_gate64_adapter_v3_20260627/checkpoints/frozen/checkpoint-0`.
- Temporary diagnostic checkpoint:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp35_minimax_flow_dpo_rescue/inference_sensitivity_20260627/temporary_perturbed_checkpoint`.
- Perturbation: deterministic scale `1.01` applied to 16 currently trainable
  MiniMax transformer tensors, saved only under the Exp35 output root.

Results:

- Identity control Step0A vs Step0B: max full MAE `0.0`; frame hashes match
  `4/4`.
- Perturbed vs Step0 mean full MAE: `0.08821829589193357`.
- Perturbed vs Step0 mean mask MAE: `0.15630244233590715`.
- Adapter scale sweep: `NOT_APPLICABLE_NO_LORA_SCOPE_IN_EXP30` because Exp30
  trained the full MiniMax transformer rather than a LoRA adapter.
- Codex visual review: opened `4/4` temporal comparison strips; `4/4` showed
  subtle nonzero response with no collapse, no black/purple failure, no
  obvious temporal artifact, and no new visible outside damage.

Interpretation:

MiniMax inference does use the transformer checkpoint weights. The Exp30
no-change failure is not an ignored-checkpoint or stale-output fallback. The
positive-control instead reinforces the previous forensic conclusion: useful
MiniMax rescue requires objective/update-scale, trainable-scope, or
bad-noise/hard-timestep changes, not blind extra steps.

Reports:

- `reports/exp35_minimax_inference_sensitivity.md`
- `reports/exp35_minimax_inference_sensitivity.csv`
- `reports/exp35_minimax_inference_sensitivity_visual_review.csv`
- `reports/exp35_minimax_inference_sensitivity_summary.json`

## 2026-06-27 Trainable-Scope Audit

- Status: `MINIMAX_TRAINABLE_SCOPE_CURRENT_OK`.
- Training performed in this milestone: false.
- GPU used: none.
- Audited checkpoint:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp30_vor_or_multimodel_minimax/minimax_gate64_adapter_v3_20260627/checkpoints/frozen/checkpoint-0`.
- Tensor count: `461`.
- Total parameter count represented by the checkpoint: `1127055424`.
- LoRA / adapter tensor count: `0`.
- Exp30 trainable scope: `all_transformer_parameters`.
- Sensitivity evidence: `MINIMAX_INFERENCE_SENSITIVITY_PASS`.

Module-family summary:

- attention q/k/v/out tensors: `60` each.
- MLP tensors: `120`, covering `826068480` parameters.
- Each MiniMax transformer block contributes about `36991232` parameters.

Conclusion:

The current Exp30 trainable scope is not too small and is not ignored by
inference. It is the full transformer scope. No expanded LoRA scope was
prepared in this milestone because the bottleneck is now better explained by
objective/update scale and missing hard-noise/timestep selection. Future rescue
recipes must still remain bounded and must not blindly extend the old
full-transformer 10-step recipe.

Reports:

- `reports/exp35_minimax_trainable_scope_audit.md`
- `reports/exp35_minimax_trainable_scope_audit.csv`
- `reports/exp35_minimax_trainable_scope_summary.json`

## 2026-06-27 Winner-SFT Positive-Control

Exp35 winner-SFT positive-control status:
`MINIMAX_POSITIVE_CONTROL_PASS_HELDOUT_QUALITY_NEGATIVE`.

This milestone ran bounded 10-step supervised winner reconstruction as a
positive-control, not DPO and not a long training run. It used PAI GPU6 only
and did not touch protected left CLI/Exp31/Exp33 processes. Because
`/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp35_minimax_flow_dpo_rescue`
was not writable by `hj`, checkpoints for this diagnostic milestone were
stored under the Exp35 autoresearch log root:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp35_minimax_flow_dpo_rescue/winner_sft_positive_control_20260627/checkpoints`

Results:

- Scope: current full MiniMax transformer (`S0_current_full_transformer`).
- Steps: `10` per recipe.
- Recipes: AdamW LR `1e-5`, `3e-5`, `1e-4`.
- All three recipes reduced train loss and produced nonzero checkpoint/output
  changes without NaN/Inf.
- Best training-loss recipe was LR `1e-5`, loss
  `0.7092440128 -> 0.0127931200`.
- Heldout mean mask PSNR deltas were negative:
  `-0.2448377791`, `-0.8897026274`, `-4.2619560566`.
- Heldout mean boundary PSNR deltas were negative:
  `-0.6611956197`, `-2.0405589461`, `-6.4308968032`.

Codex visual review:

- Opened `12/12` generated heldout temporal strips.
- LR `1e-5`: `3/4` slightly worse, `1/4` tie.
- LR `3e-5`: `3/4` clearly worse, `1/4` Pareto-mixed/tie.
- LR `1e-4`: `4/4` new artifacts, including green/purple drift, black/cyan
  blocks, blur/occlusion-like failures, and broad outside damage.

Decision:

This confirms MiniMax can be trained and that inference responds to the
updated checkpoint. It does not produce heldout quality improvement. Do not
advance to 30-step from winner-SFT, and do not write third-backbone positive
evidence from this milestone.

Reports:

- `reports/exp35_minimax_winner_sft_positive_control.md`
- `reports/exp35_minimax_winner_sft_positive_control.csv`
- `reports/exp35_minimax_winner_sft_metrics.csv`
- `reports/exp35_minimax_winner_sft_visual_review.csv`
- `reports/exp35_minimax_winner_sft_summary.json`

## 2026-06-27 Bad-Noise / Hard-Timestep Miner

Exp35 bad-noise miner status: `MINIMAX_BAD_NOISE_STATES_READY`.

This milestone mined frozen MiniMax residual states only. It did not update
model weights, did not run a recipe, did not evaluate output quality, and does
not unlock 30-step training. The miner used PAI GPU0 after confirming other
GPUs were occupied or reserved by protected lanes.

State mining details:

- Train rows: `32`.
- Heldout rows: `16`.
- Candidate states per row: `16` (`K_noise=4`, `K_timestep=4`).
- Timesteps: `0.15`, `0.35`, `0.55`, `0.75`.
- Dtype: `bfloat16`.
- CSV rows: `768`.
- Train winner-advantage mask mean: `0.053676288894166646`.
- Heldout winner-advantage mask mean: `0.030786066912696697`.
- Train state manifest SHA256:
  `fbadd0d2565c4bb49245931742215c4d074c9834b369342398058b4ed9732047`.
- Heldout state manifest SHA256:
  `947f6c0f660229f1da92cb756ee7e03cda4b2215d1ae8f154999574b590ec1fb`.

Selection policy:

- `hard_state_A`: maximum loser local residual with outside sanity filter.
- `hard_state_B`: maximum preference violation / weakest winner advantage.
- `hard_state_C`: maximum winner-risk with outside sanity filter.

Preregistered next-step options are limited to bounded 10-step recipe tests:
`H0` fixed A, `H1` online K=4 worst valid, or `H2` 50% random plus 50%
fixed A. No 30-step run is allowed until a 10-step recipe passes the quality
gate with real heldout metrics and video review.

Reports and manifests:

- `reports/exp35_minimax_bad_noise_miner.md`
- `reports/exp35_minimax_bad_noise_miner.csv`
- `reports/exp35_minimax_bad_noise_summary.json`
- `exp35_minimax_flow_dpo_rescue/manifests/train32_bad_noise_states.jsonl`
- `exp35_minimax_flow_dpo_rescue/manifests/heldout16_eval_states.jsonl`

## 2026-06-27 Rescue Recipe Preregistration

Exp35 recipe preregistration status:
`MINIMAX_RESCUE_RECIPES_PREREGISTERED`.

This milestone wrote the bounded 10-step recipe plan only. No training,
inference, metrics, video review, 30-step, long training, or RC-FPO was run.

Active recipes:

- `R1` LoVI-Linear-Frozen-HardNoise.
- `R2` LoVI-Linear-EMA-HardNoise with EMA decay `0.995`.
- `R3` WinnerAnchor-Linear-Hybrid with `lambda_winner_sft=0.05` and
  `lambda_outside=0.02`.

Fixed settings:

- Train/heldout data: locked Exp30 Gate64 V3 train32 / heldout16.
- State policy: fixed `hard_state_A`.
- Target: MiniMax flow velocity `epsilon - z0`.
- Trainable scope: `S0_current_full_transformer_from_Exp30`.
- Steps: `10` only.
- LR: `1e-5`.
- Optimizer: AdamW with betas `(0.9, 0.95)`, eps `1e-8`, weight decay
  `1e-4`, grad clip `1.0`.
- Utility scale: `10`.

R4 SDPO-safe hybrid is inactive because MiniMax flow-residual SDPO
true-model parity has not been separately validated.

The next milestone may run only the preregistered 10-step recipes. A 30-step
confirmatory run remains forbidden unless the 10-step gate returns
`MINIMAX_RESCUE_10STEP_RECIPE_PASS`.

Reports:

- `reports/exp35_minimax_rescue_recipe_preregistration.md`
- `reports/exp35_minimax_rescue_recipe_preregistration.json`

## 2026-06-27 Rescue 10-Step Recipe Gate

Exp35 rescue recipe status: `MINIMAX_RESCUE_RECIPE_NOT_READY`.

This milestone ran only the preregistered bounded 10-step rescue recipes:
`R1` LoVI-Linear-Frozen-HardNoise, `R2` LoVI-Linear-EMA-HardNoise, and `R3`
WinnerAnchor-Linear-Hybrid. It used the locked Exp30 Gate64 train32 /
heldout16 split, fixed `hard_state_A`, LR `1e-5`, utility scale `10`, and PAI
GPU0. No 30-step, long training, RC-FPO, protected-lane action, or left-side
controller action was launched.

Heldout metric deltas, Step10 minus Step0:

| Recipe | full PSNR | mask PSNR | boundary PSNR | outside PSNR | temporal-diff MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| R1 | `+0.065600` | `-0.048611` | `-0.423993` | `-0.307885` | `+0.303775` |
| R2 | `+0.057080` | `-0.053910` | `-0.434234` | `-0.321102` | `+0.305560` |
| R3 | `+0.002126` | `-0.081454` | `-0.493050` | `-0.419038` | `+0.299251` |

Codex visual review covered `48/48` heldout Step0-vs-Step10 temporal strips:

- R1: `9` tie, `5` slightly worse, `2` metric-mixed but not visibly better.
- R2: `9` tie, `5` slightly worse, `2` metric-mixed but not visibly better.
- R3: `8` tie, `6` slightly worse, `2` metric-mixed but not visibly better.

There were no collapse-level black/purple failures, but also no reliable
quality-positive rows. The three recipes produced subtle output movement,
brightness/texture drift, and negative local/boundary/outside metrics rather
than usable repair. Therefore:

- `MINIMAX_RESCUE_10STEP_RECIPE_PASS`: false.
- 30-step confirmatory MiniMax micro: not unlocked and not run.
- Third-backbone quality-positive evidence: not supported.
- MiniMax remains trainable/plumbing-positive only.

Reports:

- `reports/exp35_minimax_rescue_10step.md`
- `reports/exp35_minimax_rescue_10step_metrics.csv`
- `reports/exp35_minimax_rescue_10step_diagnostics.csv`
- `reports/exp35_minimax_rescue_10step_visual_review.csv`
- `reports/exp35_minimax_rescue_10step_summary.json`
