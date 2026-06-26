# Exp29 Continuation Readback

Date: 2026-06-26

Status: `READBACK_COMPLETED`

This readback gates the Exp29 continuation work. No GPU task was started before
reading the branch, PRD, registry, previous Exp29 reports, MiniMax/EffectErase
code pointers, and the left CLI protection state.

## Git State

- branch: `research/exp29-minimax-effecterase-adapter-feasibility-20260626`
- HEAD: `4b8d68af3ebd0f6981e697baee952b5f0e1ca76f`
- worktree: `/home/hj/H20_Video_inpainting_DPO_exp29_or_adapters`
- start status: clean
- latest local commit: `4b8d68a Run MiniMax adapter gates`
- remote fetch: completed before this readback

## Files Read

PRD:

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/49_exp29_or_adapter_feasibility.md`

Registry:

- `experiment_registry/exp29_or_adapter_feasibility/status.md`
- `experiment_registry/exp29_or_adapter_feasibility/paths.yaml`
- `experiment_registry/exp29_or_adapter_feasibility/config.yaml`
- `experiment_registry/exp29_or_adapter_feasibility/results.tsv`
- `experiment_registry/exp29_or_adapter_feasibility/metric_summary.md`
- `experiment_registry/exp29_or_adapter_feasibility/qualitative_summary.md`

Previous Exp29 reports:

- `reports/exp29_or_adapter_readback.md`
- `reports/exp29_minimax_repo_weight_audit.md`
- `reports/exp29_effecterase_repo_weight_audit.md`
- `reports/exp29_minimax_inference_smoke.md`
- `reports/exp29_minimax_inference_visual_review.csv`
- `reports/exp29_minimax_trainable_forward_audit.md`
- `reports/exp29_minimax_trainable_forward_audit.json`
- `reports/exp29_minimax_zero_gap_gate.md`
- `reports/exp29_minimax_one_step_gate.md`
- `reports/exp29_minimax_10step_micro.md`
- `reports/exp29_minimax_10step_metrics.csv`
- `reports/exp29_minimax_10step_visual_review.csv`
- `reports/exp29_minimax_adapter_gates.json`
- `reports/exp29_minimax_10step_heldout_metrics.json`
- `reports/exp29_minimax_effecterase_adapter_summary.md`
- `reports/exp29_effecterase_asset_matrix.json`
- `reports/exp29_minimax_asset_matrix.json`

Code and paper pointers read:

- MiniMax-Remover paper text snippets confirming flow interpolation and
  velocity target.
- EffectErase README and local official repository structure.
- `exp29_or_adapter_feasibility/scripts/run_minimax_adapter_gates.py`
- local MiniMax official repository code and local EffectErase repository code
  pointers from the prior audit.

## Previous MiniMax State

- repo ready: true
- weights ready: true
- inference smoke: passed technical smoke, 4/4 outputs
- trainable forward: passed
- target: `flow_velocity = epsilon - z0`
- zero-gap: passed, DPO loss approximately `log(2)`
- one-step strict reload: passed
- 10-step status: `MINIMAX_10STEP_PARETO_MIXED`
- optimizer used in successful 10-step micro: `SGD(lr=1e-7)`
- step10 parameter delta probe: `1.1061271569642785e-10`
- reference delta probe: `0.0`
- heldout metrics:
  - `davis_hockey`: PSNR delta `-0.0008006330522825067`
  - `davis_koala`: PSNR delta `+0.002413723854619576`
- visual conclusion: Step10 and Step0 are effectively tied.
- previous final MiniMax status: `MINIMAX_ADAPTER_POSSIBLE_NEEDS_MORE_WORK`

## Previous EffectErase State

- repo ready: true
- repo path: `/home/hj/video_inpainting_third_party/EffectErase`
- commit: `bcee0a5da5ef387c2ba39390dc4d579503669fb8`
- license: `CC BY-NC 4.0`
- weights ready: false
- missing assets include `EffectErase.ckpt` and Wan2.1-Fun-1.3B-InP weights.
- inference smoke: blocked
- trainable forward: blocked without weights
- previous final EffectErase status: `EFFECTERASE_BLOCKED`

## Left CLI Protection

Only read-only checks were performed.

- controller: `/home/hj/cli4_controller`
- runtime: `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/cli4`
- protected branches:
  - `research/exp25-vor-gate16-cli4-20260625`
  - `research/exp27-localdpo-objective-cli4-20260625`
  - `research/exp28-fine-inner-boundary-sweep-20260625`
- protected worktrees:
  - `/home/hj/H20_Video_inpainting_DPO_exp25_cli4`
  - `/home/hj/H20_Video_inpainting_DPO_exp27_cli4`
  - `/home/hj/H20_Video_inpainting_DPO_exp28_inner_boundary`
- signals sent to left CLI: no
- left files modified: no

Observed PAI host: `dsw-753014-85f54df947-bkp7h`

Observed left CLI processes:

- PID `258013`: `cli4_remote_monitor_5min.sh`
- PID `413869`: Exp28 pairC inner8 lane wrapper
- PID `413885`: Exp28 trial runner
- PID `536141`: Exp28 DAVIS50 eval shell
- PID `536143`: Exp28 DAVIS50 metric/eval Python

Observed left GPU use:

- GPU3: Exp28 DAVIS50 evaluation PID `536143`, about 5.6-6.6 GiB.
- GPU1/GPU2/GPU3/GPU4: reserved by CLI runtime locks and therefore not used
  by this right-side Exp29 continuation, even when idle.

Right-side eligible GPU candidates after two checks:

- GPU0
- GPU5
- GPU6
- GPU7

Maximum right-side parallelism remains 2 GPUs.

## Banned Repeats

- no VideoPainter training
- no VideoPainter 100-step
- no MiniMax long training
- no 500/1000/2000-step training
- no RC-FPO
- no VOR-Eval training or tuning
- no modification of `inference/metrics.py`
- no modification of shared trainer
- no modification of Exp1-Exp28 historical results
- no universal-adapter claim

## Pending Milestones

1. Analyze why MiniMax 10-step nearly did not change outputs.
2. Build a real medium-hard MiniMax micro preference set before any further
   MiniMax training.
3. Evaluate at most four MiniMax optimizer/objective recipes with 10-step
   budgets.
4. Run at most one MiniMax 30-step confirmatory micro if a recipe passes.
5. Recover or confirm absence of EffectErase weights.
6. Run EffectErase inference smoke only if weights are ready.
7. Audit EffectErase adapter feasibility only if inference and training forward
   are actually available.

## Current Promotion Gates

MiniMax can advance beyond the previous plumbing result only if it first
produces a medium-hard train16/heldout16 set and then a small recipe gate with
real video changes, finite losses, strict reloads, no collapse, and heldout
visual/metric evidence.

EffectErase can advance from `EFFECTERASE_BLOCKED` only after the official
weights and Wan base assets are found or recovered. Inference-only success makes
it an OR baseline/diagnostic candidate, not a true adapter success.

