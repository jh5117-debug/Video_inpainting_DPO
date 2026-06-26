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

## 2026-06-26 Continuation Readback

- Status: `EXP29_CONTINUATION_READBACK_COMPLETED`
- Branch: `research/exp29-minimax-effecterase-adapter-feasibility-20260626`
- HEAD: `4b8d68af3ebd0f6981e697baee952b5f0e1ca76f`
- Left CLI protection was re-audited read-only. The observed left runtime keeps
  GPU1/GPU2/GPU3/GPU4 reserved, with Exp28 DAVIS50 evaluation active on GPU3.
  No signal was sent and no left-side file was modified.
- Right-side eligible GPU candidates after two checks: GPU0/GPU5/GPU6/GPU7,
  max two concurrent tasks.
- MiniMax remains `MINIMAX_ADAPTER_POSSIBLE_NEEDS_MORE_WORK` pending a
  medium-hard data-quality gate and a better small optimizer/precision recipe.
- EffectErase remains `EFFECTERASE_BLOCKED` until official weights and Wan base
  assets are recovered or confirmed unavailable.

Report:

- `reports/exp29_continuation_readback.md`

## 2026-06-26 MiniMax 10-Step Failure Analysis

- Status: `MINIMAX_10STEP_FAILURE_ANALYZED`
- The previous MiniMax 10-step gate used `SGD(lr=1e-7)` after fp16 AdamW
  produced NaNs. This stabilized plumbing but produced a step10 parameter delta
  probe of only `1.1061271569642785e-10`.
- Gradients were finite, so the primary issue was not a missing backward path.
  The effective update was too small, the training losers were mostly
  trivial-bad, and the heldout set had only two rows.
- Decision: do not extend the same recipe. MiniMax must first pass a
  medium-hard train16/heldout16 data-quality gate, then a bounded optimizer /
  precision recipe gate, before any 30-step micro can run.

Reports:

- `reports/exp29_minimax_10step_failure_analysis.md`
- `reports/exp29_minimax_10step_failure_analysis.csv`
- `reports/exp29_minimax_next_micro_plan.md`

## 2026-06-26 MiniMax Preference Data Quality Gate

- Status: `MINIMAX_DATA_YIELD_INSUFFICIENT`
- Candidate generation used 32 VOR-train/search/shadow non-VOR-Eval sources,
  17 frames per source, and three pre-registered seeds per source.
- Total candidates: 96.
- Classification:
  - `MEDIUM_HARD_ELIGIBLE`: 23
  - `HARD_BUT_PLAUSIBLE`: 4
  - `TOO_CLOSE`: 3
  - `TRIVIAL_BAD`: 60
  - `TECHNICAL_INVALID`: 6
- Although 27 candidates were eligible, they came from only 9 unique scene
  groups. A scene-disjoint train16/heldout16 split is therefore impossible.
- Locked manifests:
  - train rows: 9, SHA256
    `8d9986537f04ef36a9907f093663593b5e9d87e131ed130935796d7df29cd33d`
  - heldout rows: 0, SHA256
    `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
  - rejected rows: 87, SHA256
    `38f190abc1005cbbae5a475dc3023149132aee5fa69eeaad2066d8f987ee932f`
- Decision: do not run optimizer recipe search or 30-step MiniMax micro gate
  from this data. MiniMax remains `ADAPTER_POSSIBLE_NEEDS_MORE_WORK`, not a
  third-backbone quality positive.

Reports:

- `reports/exp29_minimax_preference_data_quality.md`
- `reports/exp29_minimax_preference_data_quality.csv`
- `reports/exp29_minimax_preference_video_review.csv`
- `reports/exp29_minimax_preference_data_quality_summary.json`

## 2026-06-26 EffectErase Weight Recovery

- Status: `EFFECTERASE_WEIGHTS_READY`
- Official EffectErase and Wan2.1-Fun InP assets were downloaded on HAL and
  rsynced to the Exp29 PAI/NAS autoresearch cache.
- PAI SHA256 verification checked 19 manifest entries and all returned `OK`.
- Key recovered assets: `EffectErase.ckpt`, `Wan2.1_VAE.pth`,
  `diffusion_pytorch_model.safetensors`,
  `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`, and
  `models_t5_umt5-xxl-enc-bf16.pth`.
- Asset root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_weight_recovery_20260626`
- This milestone does not run inference smoke or adapter gates. EffectErase
  remains VOR baseline/diagnostic only until non-confounded evaluation is
  locked.

Reports:

- `reports/exp29_effecterase_weight_recovery.md`
- `reports/exp29_effecterase_weight_recovery.csv`
- `reports/exp29_effecterase_weight_recovery.json`

## 2026-06-26 Continuation V2 Readback

- Status: `EXP29_CONTINUATION_V2_READBACK_COMPLETED`
- Branch/HEAD confirmed:
  `research/exp29-minimax-effecterase-adapter-feasibility-20260626` at
  `6c97d4b74f331ce4db089224f7dcf9ec6eb283ce`.
- Re-read Exp29 PRD, registry, previous MiniMax reports, EffectErase weight
  recovery reports, and current code pointers before any GPU task.
- Left CLI was inspected read-only. Runtime locks reserve GPU1/GPU2/GPU3/GPU4
  even though PAI GPUs currently report 0 MiB and no compute process.
- No left-side signal was sent and no left-side file was modified.
- Next allowed milestones are architecture-family audit, EffectErase smoke
  pre-registration/smoke, and MiniMax expanded source-pool planning.

Report:

- `reports/exp29_continuation_v2_readback.md`
