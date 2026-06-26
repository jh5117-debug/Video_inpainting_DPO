# PRD 49: Exp29 OR Adapter Feasibility

## Objective

Exp29 audits MiniMax-Remover and EffectErase as:

1. OR baselines;
2. loser generators;
3. possible future true DPO adapter backbones.

This is a feasibility and evidence-quality track. It does not start long
training, RC-FPO, VideoPainter 100-step continuation, or third-backbone DPO
training.

## Source State

Exp26 VideoPainter LoVI-DPO completed search-dev, independent shadow-dev, and
external DAVIS-derived validation. The accepted statement is:

`CROSS_BACKBONE_ADAPTER_EVIDENCE_CONFIRMED`

The forbidden statement remains:

`UNIVERSAL_ADAPTER`

External DAVIS-derived VideoPainter validation was not confirmed, so cross-data
robustness remains open.

## Left CLI Protection

The left CLI controller and its worktrees are treated as read-only external
state. Exp29 may inspect GPU/process status but must not modify files, locks,
heartbeats, branches, worktrees, or processes associated with:

- `/home/hj/cli4_controller`
- `/home/hj/H20_Video_inpainting_DPO_exp25_cli4`
- `/home/hj/H20_Video_inpainting_DPO_exp27_cli4`
- `/home/hj/H20_Video_inpainting_DPO_exp28_inner_boundary`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/cli4`

## Initial Readback

- Branch: `research/exp29-minimax-effecterase-adapter-feasibility-20260626`
- Base: `origin/research/exp26-videopainter-dpo-v2`
- Initial base HEAD: `568a7dfb48bcdfce893176a1dd48c653414a13a8`
- PRD selected: `PRD/49_exp29_or_adapter_feasibility.md`
- New isolated code root: `exp29_or_adapter_feasibility/`
- New registry: `experiment_registry/exp29_or_adapter_feasibility/`

## Initial Local Asset Hints

- MiniMax local candidate repo:
  `/home/hj/dpo-2-1-exp/third_party_baselines/MiniMax-Remover/repo`
- EffectErase local candidate repo:
  `/home/hj/video_inpainting_third_party/EffectErase`

These are hints only. Exp29 must still perform repo, license, weight, code, and
trainable-forward audits before running any smoke.

## 2026-06-26 Repo And Weight Audit

MiniMax:

- `MINIMAX_REPO_READY`
- `MINIMAX_WEIGHTS_READY`
- Local repo: `/home/hj/dpo-2-1-exp/third_party_baselines/MiniMax-Remover/repo`
- PAI/NAS weights:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/minimax_remover/current`
- Flow target from paper/code audit: velocity / `epsilon - z0`.
- No inference smoke, trainable-forward gate, zero-gap, one-step, or 10-step
  gate has run yet.

EffectErase:

- `EFFECTERASE_REPO_READY`
- `EFFECTERASE_BLOCKED_NO_WEIGHTS`
- Local repo: `/home/hj/video_inpainting_third_party/EffectErase`
- Official inference requires `EffectErase.ckpt` and `Wan2.1-Fun-1.3B-InP`
  assets, which were not found in the audited paths.
- Generic Wan training utilities are present, but removal-specific adapter
  feasibility is blocked until official weights are available.
- VOR use remains diagnostic/baseline only because of the VOR-training
  confound recorded in Exp26.

Reports:

- `reports/exp29_minimax_repo_weight_audit.md`
- `reports/exp29_effecterase_repo_weight_audit.md`

## Promotion Gates

MiniMax can only become `MINIMAX_TRUE_ADAPTER_FEASIBILITY_CONFIRMED` after:

- verified repo and weights;
- official inference smoke with video review;
- native trainable forward;
- policy/reference zero-gap;
- one-step strict reload;
- 10-step micro gate with held-out review.

EffectErase can only become `EFFECTERASE_TRUE_ADAPTER_FEASIBILITY_CONFIRMED`
after the same technical evidence plus a non-confounded data design. If its
available evidence is VOR-trained only, it must be labeled technical or
baseline/diagnostic rather than scientific positive.

## 2026-06-26 Inference Smoke And Trainable Forward

MiniMax:

- `MINIMAX_INFERENCE_SMOKE_PASSED_WITH_VISUAL_QUALITY_RISKS`
- 4/4 fixed smoke samples ran with official local weights and produced 9
  frames each.
- Visual review:
  - `davis_bear`: medium-hard candidate; mostly successful local removal.
  - `davis_bus`: trivial-bad; large bus remains.
  - `davis_mallard-water`: trivial-bad; duck remains with blue/black artifact.
  - `davis_elephant`: trivial-bad; elephant remains with haze/smoothing.
- `MINIMAX_TRAINABLE_FORWARD_PASSED`
- Flow target implemented as `epsilon - z0`; native transformer backward
  produced finite gradients.

EffectErase:

- `EFFECTERASE_INFERENCE_SMOKE_BLOCKED_NO_WEIGHTS`
- Official repo is available, but `EffectErase.ckpt` and required Wan assets
  were not found. No fallback checkpoint was used.

Reports:

- `reports/exp29_minimax_inference_smoke.md`
- `reports/exp29_minimax_inference_visual_review.csv`
- `reports/exp29_minimax_trainable_forward_audit.md`
- `reports/exp29_effecterase_inference_smoke.md`

## 2026-06-26 Adapter Gate Decision

MiniMax:

- `MINIMAX_ZERO_GAP_PASSED`
- `MINIMAX_ONE_STEP_STRICT_RELOAD_PASSED`
- `MINIMAX_10STEP_PARETO_MIXED`
- Final status: `MINIMAX_ADAPTER_POSSIBLE_NEEDS_MORE_WORK`

The first fp16 AdamW micro attempt produced NaNs after step 1. A conservative
SGD micro-update was used as an engineering fix to verify finite update
mechanics. It completed 10 steps without NaNs and strict-loaded step1/step10,
but heldout Step10 videos were visually almost unchanged from Step0. Therefore
MiniMax is a credible next-adapter candidate, but not a confirmed third
backbone yet.

EffectErase:

- Final status: `EFFECTERASE_BLOCKED`

No EffectErase smoke or adapter gate was run because official weights were not
available.

Reports:

- `reports/exp29_minimax_zero_gap_gate.md`
- `reports/exp29_minimax_one_step_gate.md`
- `reports/exp29_minimax_10step_micro.md`
- `reports/exp29_minimax_effecterase_adapter_summary.md`
