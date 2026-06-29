# Exp42 PAI MiniMax Data Readback

Status: `EXP42_PAI_MINIMAX_DATA_READBACK_COMPLETED`

Branch: `research/exp42-pai-minimax-successful-removal-badnoise-20260629`

Start HEAD: `7dd81ef8baf1377009a4e74b022b9904e2a84957`

Base branch: `origin/research/exp40-minimax-psnr-safe-rescue-20260628`

## Files Read

PRDs:

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/50_exp30_vor_or_multimodel_minimax.md`
- `PRD/51_exp35_minimax_flow_dpo_rescue.md`
- `PRD/52_exp36_minimax_objective_rescue.md`
- `PRD/53_exp37_minimax_localdpo_badnoise_rescue.md`
- `PRD/54_exp38_minimax_full_adapter_breakthrough.md`
- `PRD/55_exp40_minimax_psnr_safe_rescue.md`

Registries:

- `experiment_registry/exp30_vor_or_multimodel_minimax/status.md`
- `experiment_registry/exp35_minimax_flow_dpo_rescue/status.md`
- `experiment_registry/exp36_minimax_objective_rescue/status.md`
- `experiment_registry/exp37_minimax_localdpo_badnoise_rescue/status.md`
- `experiment_registry/exp38_minimax_full_adapter_breakthrough/status.md`
- `experiment_registry/exp40_minimax_psnr_safe_rescue/status.md`

Reports:

- `reports/exp36_minimax_inference_sensitivity.md`
- `reports/exp36_minimax_winner_sft_positive_control.md`
- `reports/exp37_localdpo_style_or_corruption_pool.md`
- `reports/exp37_minimax_badnoise_diagnostic_scan.md`
- `reports/exp37_minimax_localdpo_badnoise_10step.md`
- `reports/exp38_minimax_failure_taxonomy.md`
- `reports/exp38_minimax_train_overfit_diagnosis.md`
- `reports/exp38_localdpo_v2_pool.md`
- `reports/exp38_minimax_badnoise_v2_diagnostic_scan.md`
- `reports/exp38_minimax_sft_dpo_rescue_10step.md`
- `reports/exp40_minimax_psnr_safe_readback.md`
- `reports/exp40_r1_sample_level_diagnosis.md`
- `reports/exp40_minimax_sft_psnr_safe_grid.md`
- `reports/exp40_minimax_paper_positioning.md`

Code and external assets read:

- `exp30_vor_or_multimodel_minimax/scripts/run_minimax_gate64_adapter_gate_v3.py`
- `exp40_minimax_psnr_safe_rescue/scripts/run_step0_baseline.py`
- `exp40_minimax_psnr_safe_rescue/scripts/run_sft_psnr_safe_grid.py`
- PAI MiniMax README and entry code under
  `/mnt/nas/hj/official_repos/MiniMax-Remover_28e12b4`

Missing-in-branch files were recorded, not fabricated:

- `PRD/56_exp41_h20_minimax_parallel_bf16.md`
- `experiment_registry/exp41_h20_minimax_parallel_bf16/status.md`
- `reports/exp41_h20_minimax_official_protocol_audit.md`
- `reports/exp41_h20_official_vs_current_visual_review.csv`

Those files may exist in H20-specific worktrees, but Exp42 did not touch the
protected H20 lane.

## PAI Readback

PAI hostname: `dsw-753014-85f54df947-bkp7h`

GPU0/GPU1 readback at PAI time `2026-06-29T09:28:28+08:00`:

- GPU0: `0 MiB / 143771 MiB`, utilization `0%`, no pmon PID.
- GPU1: `0 MiB / 143771 MiB`, utilization `0%`, no pmon PID.
- Compute-app query returned no GPU0/GPU1 process.
- No process was terminated during readback.

Official MiniMax assets on PAI:

- Repo: `/mnt/nas/hj/official_repos/MiniMax-Remover_28e12b4`
- Weights symlink:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/minimax_remover/current`
- Resolved weights:
  `/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/minimax`

The MiniMax README describes a two-stage remover with Stage2 distillation for
robust, CFG-free, fewer-step inference. The official quick-start uses
`UniPCMultistepScheduler`, `torch.float16`, `num_inference_steps=12`, and
`iterations=6`. The local pipeline code exposes mask dilation via `iterations`
and no CFG path in the project protocol.

## What MiniMax Already Ruled Out

- Not an ignored-weight inference failure: Exp36 identity replay max full MAE
  was `0.0`, while a temporary 1.01x transformer perturbation produced mean
  full/mask MAE `0.088218` / `0.156302`.
- Not total trainability failure: Exp36 winner-SFT reduced train loss, changed
  parameters, strict-reloaded checkpoints, and moved heldout outputs.
- Not pure collapse: Exp37/38 recipes generally produced subtle movement,
  ties, or mixed local changes rather than black/purple collapse.
- Not a reason to run longer blindly: Exp30, Exp35, Exp36, Exp37, Exp38, and
  Exp40 all failed quality-positive heldout gates.

## Why Direct SFT/DPO Failed So Far

The strongest current explanation is data/objective/update-localization rather
than code loading:

- Exp30 frozen/EMA Linear-DPO: heldout visual better `0/32`.
- Exp35 hard-noise recipes: visual better `0/48`.
- Exp36 winner-SFT: visual better `0/24`; high-LR S0 created artifacts.
- Exp37 LocalDPO-badnoise: each recipe visual better `1/16`, below gate.
- Exp38 R1 produced weak full/mask metric movement but boundary/outside and
  visual gates failed.
- Exp40 PSNR-safe SFT grid was aggregate-negative on search. The best recipe
  still had full/mask/boundary/outside deltas
  `-1.816781` / `-1.634597` / `-1.899575` / `-2.624405`.

The missing signal is not just "more steps"; it is MiniMax-native data:
successful removal states, comparable failure states, and noise/timestep cases
that reveal where MiniMax itself succeeds or fails.

## Stage2-Style Successful-Removal Distillation

Exp42 interprets Stage2-style data construction as:

- run official MiniMax inference on VOR-Train-derived sources only;
- identify raw outputs that are visually/quantitatively acceptable removals;
- use those successful removals as `pseudo_success` targets only when they
  pass visual/metric gates;
- keep GT `V_bg` as the clean distillation target where appropriate;
- pair successful and failed MiniMax removals from the same source/noise family
  for failure-avoidance preferences;
- preserve outside/background rigorously and reject fogging, over-erasure,
  boundary destruction, hard comp, and VOR-Eval leakage.

## Available Sources and Manifests

- Exp30 Gate64 V3:
  - `exp30_vor_or_multimodel_minimax/manifests/vor_or_gate64_train32_v3.jsonl`
  - `exp30_vor_or_multimodel_minimax/manifests/vor_or_gate64_heldout16_v3.jsonl`
- Exp37 LocalDPO-style pool:
  - `exp37_minimax_localdpo_badnoise_rescue/manifests/localdpo_or_train32.jsonl`
  - `exp37_minimax_localdpo_badnoise_rescue/manifests/localdpo_or_heldout16.jsonl`
- Exp38 filtered LocalDPO v2:
  - `exp38_minimax_full_adapter_breakthrough/manifests/localdpo_v2_train30_filtered.jsonl`
  - `exp38_minimax_full_adapter_breakthrough/manifests/localdpo_v2_heldout13_filtered.jsonl`
  - `exp38_minimax_full_adapter_breakthrough/manifests/badnoise_v2_train30_states.jsonl`
- Exp40 LocalDPO v3 minimum pool:
  - `exp40_minimax_psnr_safe_rescue/manifests/exp40_localdpo_v3_train96.jsonl`
    currently contains the selected train64 minimum pool.
  - `exp40_minimax_psnr_safe_rescue/manifests/exp40_localdpo_v3_search32.jsonl`
    currently contains the selected search24 minimum pool.
  - `exp40_minimax_psnr_safe_rescue/manifests/exp40_localdpo_v3_shadow32.jsonl`
    currently contains the selected shadow24 minimum pool.

HAL-side path checks for PAI/NAS paths are not authoritative because these
manifests reference `/mnt/nas` runtime assets. PAI checks must be repeated
before mining.

## What PAI Does Differently from H20

PAI Exp42 owns MiniMax data mining and bad-noise signal construction:

- GPU0: official MiniMax inference / successful-removal mining.
- GPU1: metrics, bad-noise scan, visual pack, and short gated training only if
  data gates pass.

H20 is treated as protected and read-only. Exp42 does not change H20 runners,
does not consume H20 GPUs, and does not modify Exp41 outputs.

## Banned Repeats

- Do not rerun Exp38/Exp40 R1/R2/R3 recipes as-is.
- Do not extend failed PSNR-safe SFT recipes to 100/300/500 steps.
- Do not train before successful-removal, success/failure, and Stage2 data
  gates pass.
- Do not use VOR-Eval for training, selection, or threshold tuning.
- Do not use hard comp as the primary evaluation output.

## Success Gates

Milestone A successful-removal mining:

- successful candidates >= 24;
- medium-hard failures >= 24;
- outside damage low;
- technical-valid rate >= 95%.

Milestone B bad-noise v3:

- at least 24 usable success/failure pairs;
- hard-state local/random gradient ratio >= 1.5;
- bounded outside risk and bounded winner risk.

Milestone C Stage2 data:

- train >= 32, search >= 16, shadow >= 16;
- scene-group disjoint splits;
- all paths exist and decode;
- masks non-empty;
- VOR-Eval used = false.

Only after those gates may Exp42 run short SFT, and only if SFT passes may DPO
after Stage2 SFT run. No 500/1000/2000-step training is authorized here.

## Decision

Proceed to Milestone A after this readback is committed and pushed.

Current Exp42 status: `EXP42_PAI_MINIMAX_DATA_READBACK_COMPLETED`.
