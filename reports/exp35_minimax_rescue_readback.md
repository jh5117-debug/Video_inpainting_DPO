# Exp35 MiniMax Flow-DPO Rescue Readback

Status: `EXP35_READBACK_COMPLETED`

## Git And Scope

- Branch: `research/exp35-minimax-flow-dpo-rescue-20260627`
- Base: `origin/research/exp30-vor-or-multimodel-minimax-adapter-20260627`
- Start HEAD: `f69688fe4ff96c4d4f0dcd308eef69822fc1035b`
- Worktree: `/home/hj/H20_Video_inpainting_DPO_exp35_minimax_rescue`
- Output root: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp35_minimax_flow_dpo_rescue`
- Log root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp35_minimax_flow_dpo_rescue`

This readback created only the Exp35 scaffold. It did not run GPU inference,
training, 30-step, long training, or RC-FPO.

## Files Read

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/50_exp30_vor_or_multimodel_minimax.md`
- `experiment_registry/exp30_vor_or_multimodel_minimax/status.md`
- `experiment_registry/exp30_vor_or_multimodel_minimax/results.tsv`
- `experiment_registry/exp30_vor_or_multimodel_minimax/metric_summary.md`
- `experiment_registry/exp30_vor_or_multimodel_minimax/qualitative_summary.md`
- `reports/exp30_vor_or_gate64_multimodel_pool_v3.md`
- `reports/exp30_vor_or_gate64_multimodel_summary_v3.json`
- `reports/exp30_minimax_gate64_adapter_zero_gap_v3.md`
- `reports/exp30_minimax_gate64_adapter_one_step_v3.md`
- `reports/exp30_minimax_gate64_adapter_10step_v3.md`
- `reports/exp30_minimax_gate64_adapter_10step_metrics_v3.csv`
- `reports/exp30_minimax_gate64_adapter_10step_visual_review_v3.csv`
- `reports/exp30_minimax_gate64_adapter_diagnostics_v3.csv`
- `reports/exp30_minimax_gate64_adapter_summary_v3.json`
- `exp30_vor_or_multimodel_minimax/scripts/run_minimax_gate64_adapter_gate_v3.py`

## Readback Answers

1. What failed in Exp30 MiniMax 10-step?

The MiniMax 10-step gate failed to produce visible or metric-positive heldout
quality change. Frozen and EMA recipes both had negative mean mask, boundary,
and outside PSNR deltas, and visual review found Step10 better `0/32` with
`32/32` ties/no visible improvement.

2. Was the failure data-related, plumbing-related, or recipe-related?

It is currently recipe/update-state related. Gate64 V3 data is ready and
scene-disjoint. Zero-gap, one-step strict reload, checkpoint save/reload, and
Step10 inference completed. The problem is that the update was too weak or in
the wrong objective/noise regime to move rendered outputs usefully.

3. Which trainable modules were used in Exp30?

Exp30 did not use a restricted LoRA scope. The script set
`requires_grad_(True)` on every parameter of `Transformer3DModel`, while the
VAE and reference were frozen except EMA updates in the EMA recipe.

4. What was the parameter delta after 10 steps?

The script's delta probe reported very small movement: frozen Step10 delta
probe `2.743745512179263e-07`; EMA Step10 delta probe
`2.735606286011216e-07`. Reference delta was `0.0` for frozen and
`8.351530297171276e-09` for EMA.

5. Was the adapted checkpoint actually used by inference?

Yes according to the Exp30 script and reports. Step10 inference loaded
`checkpoint-10` into a fresh float16 MiniMax transformer copy. This still needs
a deeper Exp35 checkpoint-load/output-diff audit, but readback does not show a
silent Step0 fallback.

6. Did Step10 output differ pixel-wise from Step0?

The aggregate metrics imply nonzero but tiny pixel-level differences. The
Exp30 visual result was no visible improvement; Exp35 Milestone A must compute
explicit pixel, mask, affected, and outside diffs.

7. Did loss decrease while output stayed unchanged?

The prior diagnostic rows used different train rows and timesteps per step, so
loss is not a clean monotonic curve. Utility stayed near `0.5`, margins were
tiny, and rendered outputs stayed visually tied. Exp35 must audit loss/utility
scale directly.

8. Was flow target `epsilon - z0` used?

Yes. The Exp30 script constructs `zt = t * epsilon + (1 - t) * z0` and uses
target velocity `epsilon - z0`.

9. Was bad-noise sampling used?

No. Exp30 used one deterministic seed/timestep per step. It did not perform
K-noise/K-timestep mining, minimax bad-noise selection, or hard-state
selection.

10. Was utility scale too weak?

Likely. Linear utility stayed near `0.5`, beta was `1.0`, and LR was `5e-7`.
This is a primary hypothesis for Milestone A/F, not a final root-cause claim
until audited.

11. Was learning rate too small?

Likely. Full-transformer trainable scope plus LR `5e-7` yielded Step10 delta
only around `2.7e-7`. Exp35 must test this with positive-control and bounded
recipe gates rather than blindly increasing steps.

12. Was LoRA scope too small or ignored?

No LoRA scope was used in Exp30. The risk is not an ignored LoRA adapter in
the previous gate; the risk is full-model updates being numerically too small
or objective/noise-state mis-scaled. Exp35 may prepare explicit LoRA scopes
only after sensitivity and scope audits.

13. What is this round allowed to change?

Exp35 may audit no-change behavior, test reversible inference sensitivity,
prepare isolated trainable scopes, run a tiny winner-SFT positive-control,
mine bad-noise/hard-timestep states, preregister bounded rescue recipes, and
run at most 10-step rescue recipes. 30-step is conditional on a 10-step recipe
pass.

14. What is forbidden?

Forbidden: touching Exp31/Exp33/cli4, modifying `inference/metrics.py` or
shared trainers, modifying Exp1-Exp29 history, overwriting Exp30 outputs,
running 500/1000/2000-step or RC-FPO, running 30-step before an explicit
10-step pass, and writing universal-adapter/all-models/final-SOTA claims.

## Protected Lanes

PAI readback observed:

- Exp31 VideoPainter 2000-step on GPU1, PID `1215136`, reserved.
- cli4 lock files reserve GPU1/GPU2/GPU3/GPU4.
- A non-project root process was visible in `ps` but not in compute-GPU query;
  it was not touched.

Signals sent: no.
Protected files modified: no.

## Initial Hypothesis

Allowed root-cause hypotheses entering Milestone A:

- `MINIMAX_NOCHANGE_CAUSE_LR_TOO_SMALL`
- `MINIMAX_NOCHANGE_CAUSE_UTILITY_SCALE_TOO_WEAK`
- `MINIMAX_NOCHANGE_CAUSE_TIMESTEP_TOO_EASY`
- `MINIMAX_NOCHANGE_CAUSE_OUTPUT_INSENSITIVE`
- `MINIMAX_NOCHANGE_CAUSE_UNCLEAR`

Checkpoint load and trainable-scope causes are not ruled out until Milestone A
and Milestone B complete.

