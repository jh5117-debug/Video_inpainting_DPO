# Exp29 Continuation V3 Readback

Date: 2026-06-26

Status: `EXP29_CONTINUATION_V3_READBACK_COMPLETED`

## Git

- Branch: `research/exp29-minimax-effecterase-adapter-feasibility-20260626`
- HEAD at readback: `972deab321a518638102a1ace6ed87a13456a261`
- Worktree status: clean before this readback report.
- Latest commit: `972deab Preregister EffectErase inference smoke`

## Files Read

PRD files:

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/49_exp29_or_adapter_feasibility.md`

Registry files:

- `experiment_registry/exp29_or_adapter_feasibility/status.md`
- `experiment_registry/exp29_or_adapter_feasibility/paths.yaml`
- `experiment_registry/exp29_or_adapter_feasibility/config.yaml`
- `experiment_registry/exp29_or_adapter_feasibility/results.tsv`
- `experiment_registry/exp29_or_adapter_feasibility/metric_summary.md`
- `experiment_registry/exp29_or_adapter_feasibility/qualitative_summary.md`

Reports:

- `reports/exp29_continuation_v2_readback.md`
- `reports/exp29_architecture_family_audit.md`
- `reports/exp29_architecture_family_audit.csv`
- `reports/exp29_effecterase_weight_recovery.md`
- `reports/exp29_effecterase_weight_recovery.csv`
- `reports/exp29_effecterase_weight_recovery.json`
- `reports/exp29_effecterase_smoke_preregistration.md`
- `reports/exp29_effecterase_smoke_preregistration.json`
- `reports/exp29_minimax_preference_data_quality.md`
- `reports/exp29_minimax_preference_data_quality.csv`
- `reports/exp29_minimax_preference_video_review.csv`
- `reports/exp29_minimax_preference_data_quality_summary.json`
- `reports/exp29_minimax_effecterase_adapter_summary.md`

Code and assets:

- EffectErase official repo source copied for runtime under the Exp29
  autoresearch cache.
- EffectErase official scripts and `examples/remove_wan/infer_remove_wan.py`
  were previously inspected.
- MiniMax and current Exp29 adapter scripts remain read-only for this readback.

## EffectErase State

- `EFFECTERASE_WEIGHTS_READY`
- `EFFECTERASE_SMOKE_PREREGISTERED`
- Smoke manifest:
  `exp29_or_adapter_feasibility/manifests/effecterase_smoke_preregistered.jsonl`
- Manifest SHA256:
  `54fd62a97fa69f2f17590488136d426cee77de0ed02548c46a83d8818be2b137`
- Rows: 6
- Balance: REAL/BLENDER = 3/3; small/medium/large masks = 2/2/2.
- Frames: 17, resolution 832x480, seed 2025, CFG 1.0, 50 steps.
- All rows are `diagnostic_only_vor_confounded`,
  `eligible_for_training=false`, and `vor_eval=false`.
- No EffectErase inference output has been generated yet.

Current environment note:

- A dedicated Exp29 EffectErase venv was started after preregistration.
- `torch`, `diffusers`, `transformers`, `decord`, `cv2`, `imageio`,
  `safetensors`, `accelerate`, and `modelscope` import in that venv.
- Official `infer_remove_wan.py` still needs command dry-run validation because
  the previous interrupted import reached a Qwen2.5-VL transformer symbol
  mismatch. This is a command/environment issue, not an inference result.

## MiniMax State

- Repo and weights ready.
- Inference smoke passed with visual quality risks.
- Trainable forward passed with native flow target `epsilon - z0`.
- Zero-gap and one-step strict reload passed.
- Previous 10-step: `MINIMAX_10STEP_PARETO_MIXED`.
- Current data-yield gate: `MINIMAX_DATA_YIELD_INSUFFICIENT`.
- Prior candidate counts: 96 total, 23 medium-hard, 4 hard-plausible, 3
  too-close, 60 trivial-bad, 6 technical-invalid.
- Eligible candidates: 27, but only 9 unique eligible scene groups.
- Recipe search and 30-step micro remain forbidden until an expanded
  source-pool data-yield gate passes.

## Left CLI Protection

PAI hostname: `dsw-753014-85f54df947-bkp7h`

Read-only GPU/process check at readback:

- All GPUs reported 0 MiB and 0% utilization.
- No compute applications were reported by `nvidia-smi`.
- Left CLI runtime locks still reserve GPU1/GPU2/GPU3/GPU4.
- Observed left monitor process:
  `hj 258013 ... bash /mnt/nas/hj/H20_Video_inpainting_DPO/runtime/cli4/cli4_remote_monitor_5min.sh`

Right-side eligible GPU candidates for later tasks are GPU0/GPU5/GPU6/GPU7
only, subject to a fresh two-pass check before GPU inference. GPU1-GPU4 remain
reserved for left CLI even if idle.

No signal was sent to any left process. No left runtime, worktree, branch,
heartbeat, lock, or output file was modified.

## This Round Milestones

1. EffectErase smoke input materialization check.
2. EffectErase official command dry-run.
3. EffectErase official inference smoke if inputs and command are ready.
4. EffectErase trainable-forward audit only if smoke is technically valid.
5. MiniMax expanded source-pool plan.
6. Optional MiniMax expanded candidate generation first pass after the plan is
   committed.

## Forbidden Repeats And Claims

Do not run VideoPainter continuation, MiniMax recipe/30-step, long training,
RC-FPO, or any Exp1-Exp28 edits. Do not modify `inference/metrics.py` or shared
trainer code.

Do not claim:

- `UNIVERSAL_ADAPTER`
- `ALL_MODELS_SUPPORTED`
- `FINAL_SOTA`
- `TOP_CONFERENCE_NOVELTY_CONFIRMED`
- `EFFECTERASE_TRUE_ADAPTER_FEASIBILITY_CONFIRMED` from inference smoke alone
- MiniMax third-backbone quality positive before a real data-yield and micro
  quality gate
