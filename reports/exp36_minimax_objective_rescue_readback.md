# Exp36 MiniMax Objective Rescue Readback

Status: `EXP36_READBACK_COMPLETED`

## Git And Scope

- Branch: `research/exp36-minimax-objective-rescue-20260627`
- Base: `origin/research/exp35-minimax-flow-dpo-rescue-20260627`
- Start HEAD: `fb70266d53f5f9abd5e8d09ef9d2de324a10b7d6`
- Worktree: `/home/hj/H20_Video_inpainting_DPO_exp36_minimax_objective_rescue`
- Output root: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp36_minimax_objective_rescue`
- Log root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp36_minimax_objective_rescue`

This readback created only the Exp36 scaffold and registry. It did not launch
GPU inference, training, 30-step, long training, RC-FPO, or protected-lane
actions.

## Files Read

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/50_exp30_vor_or_multimodel_minimax.md`
- `PRD/51_exp35_minimax_flow_dpo_rescue.md`
- `experiment_registry/exp30_vor_or_multimodel_minimax/status.md`
- `experiment_registry/exp35_minimax_flow_dpo_rescue/status.md`
- `reports/exp30_minimax_gate64_adapter_zero_gap_v3.md`
- `reports/exp30_minimax_gate64_adapter_one_step_v3.md`
- `reports/exp30_minimax_gate64_adapter_10step_v3.md`
- `reports/exp30_minimax_gate64_adapter_10step_metrics_v3.csv`
- `reports/exp30_minimax_gate64_adapter_10step_visual_review_v3.csv`
- `reports/exp30_minimax_gate64_adapter_diagnostics_v3.csv`
- `reports/exp35_minimax_rescue_10step.md`
- `reports/exp35_minimax_rescue_10step_metrics.csv`
- `reports/exp35_minimax_rescue_10step_diagnostics.csv`
- `reports/exp35_minimax_rescue_10step_visual_review.csv`
- `reports/exp35_minimax_rescue_10step_summary.json`
- `exp30_vor_or_multimodel_minimax/scripts/run_minimax_gate64_adapter_gate_v3.py`
- `exp35_minimax_flow_dpo_rescue/scripts/audit_exp30_minimax_nochange.py`
- `exp35_minimax_flow_dpo_rescue/scripts/run_minimax_inference_sensitivity.py`
- `exp35_minimax_flow_dpo_rescue/scripts/audit_minimax_trainable_scope.py`
- `exp35_minimax_flow_dpo_rescue/scripts/run_minimax_winner_sft_positive_control.py`
- `exp35_minimax_flow_dpo_rescue/scripts/mine_minimax_bad_noise_states.py`
- `exp35_minimax_flow_dpo_rescue/scripts/run_minimax_rescue_10step_recipes.py`

## Protected-Lane Readback

Read-only PAI check observed:

- PAI host: `dsw-753014-85f54df947-bkp7h`
- User: `hj`
- cli4 monitor process observed.
- cli4 runtime locks under
  `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/cli4`.
- Exp31 runtime locks under
  `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/idle_gpu_parallel_20260627`.
- Exp33 runtime locks under the same idle-GPU runtime root.

GPU readback at the time of this report:

- GPU0: physically free in `nvidia-smi`, not claimed by the observed cli4 /
  Exp31 / Exp33 lock names.
- GPU1: physically free but reserved by cli4 / Exp31 lock state.
- GPU2-GPU4: occupied and/or cli4-reserved.
- GPU5-GPU7: occupied by unknown/root-side compute processes.

Signals sent: no.
Protected files modified: no.
GPU reset / pkill / killall: no.

## Readback Answers

1. Which MiniMax recipes have already failed?

Exp30 failed frozen-reference and EMA Linear-DPO 10-step recipes. Exp35 failed
R1 LoVI-Linear-Frozen-HardNoise, R2 LoVI-Linear-EMA-HardNoise, and R3
WinnerAnchor-Linear-Hybrid.

2. Did they fail by collapse, no-change, or slight degradation?

They failed by no useful quality improvement plus slight degradation, not
collapse. Exp35 R1/R2/R3 had `0` visual-better rows across `48` reviewed
heldout strips and no black/purple collapse.

3. What data pool was used?

The locked Exp30 Gate64 V3 train32 / heldout16 pool:

- train32 SHA256:
  `1eda205d2dc48714269f30eb390d959549387a778e6438267e6aba087ba14196`
- heldout16 SHA256:
  `84c231ded930d740bf299b27c2a6b1e95d7decdb3665051371c5df90ae9f2ade`

4. Which objective was used?

Exp30 used MiniMax flow target `epsilon - z0` with Linear-DPO frozen/EMA
recipes. Exp35 used hard-state Linear-DPO variants plus a winner-anchor
hybrid. R4 SDPO-safe was explicitly inactive because MiniMax true-model SDPO
geometry was not yet validated.

5. Which trainable modules were used?

The current path used the full MiniMax transformer scope, not a LoRA-only
adapter. Exp35 audited `461` checkpoint tensors representing
`1,127,055,424` parameters and `0` LoRA/adapter tensors.

6. What was parameter delta?

Exp35 no-change audit found Exp30 parameter delta / param norm ratios around
`5.64e-06`. Exp35 winner-SFT positive-control produced larger nonzero deltas
and proved trainability, but heldout quality remained negative.

7. Did inference use the trained weights?

Current evidence says yes. Exp35 inference-sensitivity identity replay was
exactly deterministic, while an Exp35-only `1.01x` perturbation of 16 MiniMax
transformer tensors produced nonzero output movement without collapse.

8. Did Step10 output differ from Step0?

Yes, but not usefully. Exp35/Exp30 outputs were not byte-identical and showed
nonzero pixel movement, yet visual review found no quality-positive rows.

9. Did loss decrease while video stayed same?

Winner-SFT loss decreased strongly and outputs moved, but heldout quality
degraded. DPO/Linear utility stayed near constant around `0.5`, consistent
with objective/update-scale weakness.

10. What remains untested?

- MiniMax-specific true SDPO safe-lambda gradient geometry.
- LocalDPO-style local corruption / region-restricted objective on MiniMax.
- Whether a stricter winner-preserving objective can improve heldout quality.
- Whether S1/S2 LoRA-style scopes help after explicit forward-usage tests.
- Any 30-step confirmatory run; this remains locked until an Exp36 10-step
  pass.

## Next Gate

The next eligible milestone is the no-change forensic audit using prior
Exp30/Exp35 checkpoints and logs. It must not launch new training.

