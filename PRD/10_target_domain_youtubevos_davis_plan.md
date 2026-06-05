## 2026-06-05 Exp8 DAVIS Region-Loss Diagnostic

Exp8 now has an explicit PAI-only manual launcher:

```text
scripts/launch_exp8_d3_comp_regionloss_s1s2_2000_davis_pai.sh
```

Required target-domain policy:

- YouTube-VOS/D3 train or inference must use SFT-48000 DiffuEraser, not a naked base.
- DAVIS partial-mask validation must use ProPainter prior.
- Metrics must go through `tools/run_inpainting_metric_eval.py` and `inference/metrics.py`.
- VBench is only for video generation/full-mask prompt generation, not this inpainting task.

Exp8 validation compares:

1. SFT-48000 DiffuEraser baseline on DAVIS.
2. DPO-Stage1 + frozen SFT-Stage2 hybrid on DAVIS.
3. DPO-Stage1 + DPO-Stage2 on DAVIS.

The Stage1 hybrid must be built with `tools/build_diffueraser_dpoS1_sftS2_hybrid.py`; Stage1 2D weights must not be evaluated as a full video model directly.

## 2026-06-04 PAI Audit Backfill / Registry Correction

This update uses the returned PAI audit archive `pai_experiment_registry_reports.tar.gz` and `reports/pai_experiment_registry_paths.csv`.

Key corrections:

- Exp5 is split into three evidence units: Old Exp5 beta500 plain collapse, Exp5 beta10 plain collapse, and New Exp5 winner-gap/lose025 improvement.
- New Exp6 remains D2 no-comp + the New Exp5 winner-preserving loss. It is not plain Exp6.
- PAI audit found remote `dpo_diagnostics.csv` paths for Old Exp5, Exp5 beta10 plain, New Exp5, Exp7, Exp8, and Exp9-comp. These are marked `REMOTE_DIAG_PATH_FOUND` until the actual CSVs are fetched or summarized on PAI.
- Local H20 diagnostic snapshots exist for New Exp6, Exp9 D3-nocomp, and Exp9 D3 no-lose.
- Exp4 still has no artifact path in the audit; it remains a deleted/failed fullmask quality gate.
- YouTube-VOS/D3 work must use SFT-48000 DiffuEraser, not a naked base. H20 confirmed path: `/home/nvme01/H20_Video_inpainting_DPO/weights/diffuEraser/converted_weights_step48000`; PAI required path: `/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000`.
- Partial-mask inpainting should use ProPainter prior; fullmask/video-generation bridge runs should not rely on that prior.
- Exp7 must be repaired with 15%-20% masks and ProPainter prior before expanding D3/YouTube-VOS DPO.

Evidence files:

- `experiment_registry/experiment_matrix.md`
- `reports/pai_experiment_registry_audit_summary.md`
- `reports/all_experiments_dpo_diag_summary.md`
- `reports/exp7_prior_mask_audit.md`

## 2026-06-04 Exp7 Small-Mask / Prior Reset

This update supersedes any plan to expand D3/YouTube-VOS DPO before Exp7 is fixed.

Required facts:

- YouTube-VOS / D3 work must use the fine-tuned SFT-48000 DiffuEraser weights, not an ordinary base.
  - PAI path: `/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000`
  - HAL local reference: `/home/hj/Video_inpainting_DPO/weights/diffuEraser/converted_weights_step48000`
  - H20 confirmed path: `/home/nvme01/H20_Video_inpainting_DPO/weights/diffuEraser/converted_weights_step48000`
- Every experiment must have an `experiment_registry/<exp>/` folder with config, commands, paths, status, metric, qualitative, and dpo_diag notes.
- Every DPO experiment needs `dpo_diagnostics.csv`; missing diagnostics are marked `MISSING_DPO_DIAG`.
- Current lineage is: Exp4 failed fullmask data gate; Old Exp5 plain DPO collapse; New Exp5 comp + winner anchoring improves but is not final; New Exp6 no-comp + same winner anchoring is not plain Exp6; Exp7 changed task to partial mask but is suspicious; Exp8/Exp9 are target-domain diagnostics only.
- Fullmask/video-generation settings should not rely on ProPainter prior. Partial-mask video inpainting should use DiffuEraser/OR ProPainter prior.
- VideoDPO partial-mask masks should be reduced to 15%-20% for Exp7-fix; do not continue old large-mask D2 for new Exp7 gates.
- Train may use YouTube-VOS, but final target eval should use DAVIS at `/mnt/workspace/hj/nas_hj/data/external/davis_432_240`.

Current Exp7 audit:

- `tools/eval_generated_loser_partialmask_model.py` appears to use a direct pipeline path with masked winner frames and masks, without `inference/run_OR.py` or `--propainter_model_dir`.
- This makes the current Exp7 eval likely no-prior and may explain why SFT/base videos also look bad.
- Report: `reports/exp7_prior_mask_audit.md`.

Next planned gates after small-mask/prior data exists:

- PAI: `exp07_fix_smallmask_prior_wingap_lose025_stage1_gate1500`.
- H20: `exp07_fix_smallmask_prior_wingap_nolose_stage1_gate1000`.

Do not run DPO Stage2, VBench for inpainting, or long D3 sweeps until this sanity gate is reviewed.

# Target-Domain YouTube-VOS / DAVIS Plan

## 2026-06-04 Experiment Registry / DPO-Diag Audit Update

A formal `experiment_registry/` has been added as the source-of-truth artifact ledger. Each experiment now has an independent folder with README, paths, commands, dpo_diag, metrics, qualitative, and status notes.

Naming corrections:

- Old Exp5 and New Exp5 are separate rows and must not be merged.
- New Exp6 is the D2 no-comp + winner-anchored-loss diagnostic, not a plain Exp6.
- Exp4 is a failed full-mask generated-loser quality gate / diagnostic-only artifact.
- Exp8 is a target-domain D3 region-loss gate; it is not a completed VideoDPO bridge success claim.
- Exp9 is the YouTube-VOS/D3 target-domain partial-mask gate family.

Evidence rule:

- Every DPO experiment must have `dpo_diagnostics.csv` or be explicitly marked `MISSING_DIAG`.
- Current conclusions must be supported by metric + qualitative + dpo_diag together, or marked as incomplete / diagnostic-only.
- The aggregate dpo summary is generated by `tools/collect_dpo_diag_summaries.py` and written to `reports/all_experiments_dpo_diag_summary.md`.

Current audit status:

- H20 dpo_diag snapshots found: New Exp6 Stage1/Stage2, Exp9 D3-nocomp Stage1, Exp9 D3-comp no-lose Stage1.
- Missing / pending PAI dpo_diag: historical Exp1-3, Exp4, Old Exp5, New Exp5, Exp7, Exp7 hybrid, Exp8, Exp9 D3-comp clean gate.
- PAI must be scanned manually with the fixed-glob audit command; do not recursively scan the whole NAS.


## Domain Boundary

Final target domain:

- YouTube-VOS
- DAVIS

VideoDPO is now treated as a bridge domain only. Its role is to validate native
VideoDPO repository integration, DiffuEraser loading, generated-loser manifests,
partial-mask plumbing, winner-gap regularization, DPO diagnostics, and
Stage1/Stage2 weight loading. VideoDPO partial-mask evaluation is diagnostic
only and must not be reported as final quality.

Do not do a VideoDPO partial-mask SFT warmup. If SFT is needed, it should be on
target-domain data or closely related inpainting data.

## Current Interpretation

- Exp3 validates replacing VC2 with DiffuEraser in the native VideoDPO repo.
- Exp5 and Exp6 validate generated-loser data-only DPO and stabilization
  tricks. New Exp5 winner-anchor improves optimization stability but is not a
  final visual success.
- Exp7 validates partial-mask task support inside the native
  VideoDPO-DiffuEraser pipeline. Its VideoDPO partial-mask eval is a diagnostic,
  not final success.
- The current priority is target-domain evaluation of existing checkpoints on
  YouTube-VOS / DAVIS.

## D3 Target-Domain Data

H20 D3 root:

```text
/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4
```

PAI target root:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4
```

H20 audit from 2026-06-02:

- size: 249G
- files: 1,819,879
- shards: 3,327
- `selected_primary_comp.jsonl`: 3,327 rows
- `selected_primary_nocomp.jsonl`: 3,327 rows
- sampled 100 selected-primary-comp rows: all status `OK`
- sampled win/mask/final loser paths: 16 frames, 512x320, readable
- all sampled manifest paths are H20-only `/home/nvme01/...` absolute paths

Therefore D3 must be path-rewritten on PAI before training. Do not train
directly from the original H20-path manifests.

## D3 Sync Strategy

Use PAI-side pull from H20:

```bash
bash scripts/sync_d3_from_h20_to_pai.sh
```

Default `SYNC_MODE=slim` syncs only selected-primary manifests and the selected
primary paths required for first Exp9 training. If slim packaging proves risky,
run:

```bash
SYNC_MODE=full bash scripts/sync_d3_from_h20_to_pai.sh
```

The sync script uses `--partial --append-verify`, never uses `--delete`, and
writes logs under `logs/data_sync/` when launched through `nohup`.

After sync:

```bash
python tools/d3_post_generation_audit_and_repair.py
python tools/d3_training_readiness_check.py
```

Expected reports:

```text
<D3_ROOT>/reports/d3_post_generation_audit.md
<D3_ROOT>/reports/d3_training_readiness_report.md
```

Expected repaired manifests:

```text
manifests/selected_primary_comp.repaired.jsonl
manifests/selected_primary_nocomp.repaired.jsonl
manifests/selected_primary_comp.repaired.pai_paths.jsonl
manifests/selected_primary_nocomp.repaired.pai_paths.jsonl
```

Readiness split:

- D3 full readiness can remain false when the PAI slim sync omits secondary
  manifests.
- D3 primary-comp gate readiness is the relevant condition for the first Exp9
  Stage1 gate.
- Use `tools/d3_primary_comp_gate_readiness_check.py` to confirm
  `ready_primary_comp_gate=true`; do not let secondary-manifest absence block
  the selected-primary-comp gate.

## Target-Domain Eval Gate

Run preflight first:

```bash
bash scripts/eval_target_youtubevos_davis_checkpoints_metricpy.sh
```

The target eval must use the previous DiffuEraser best settings:

- denoise steps = 6
- no PCM
- no Gaussian blur
- no unnecessary mask dilation
- frame-wise output / metric path
- hard comp outside mask

If the current eval backend cannot guarantee these settings, write a report and
do not pretend the eval is valid.

Metric backend:

- Use VBench only for video generation / full-mask prompt generation.
- Use the project metric module for YouTube-VOS / DAVIS partial-mask video
  inpainting.
- No exact `metric.py` exists in this checkout; the existing metric module is
  `inference/metrics.py`.
- `tools/run_inpainting_metric_eval.py` is the thin adapter that imports the
  metric module and writes `summary.csv`, `summary.json`, and `summary.md`.
- Do not reimplement PSNR, SSIM, LPIPS, or temporal metrics in target eval
  scripts.

Compare existing checkpoints:

- DiffuEraser-base / current best SFT DiffuEraser
- Exp3 official_videodpo_diffueraser checkpoint
- new Exp5 winner-anchored DPO checkpoint
- new Exp6 winner-anchored DPO checkpoint if available
- Exp7 DPO-S1 + DPO-S2
- DPO-S1 + SFT-S2 hybrid

Report:

```text
reports/target_domain_youtubevos_davis_eval_report.md
```

The report must answer whether any VideoDPO-bridge DPO checkpoint transfers to
YouTube-VOS / DAVIS and whether DPO-S1 + SFT-S2 beats DPO-S1 + DPO-S2.

## Exp9 Plan Boundary

Do not start Exp9 automatically. Exp9 starts only if D3 selected-primary
readiness is true and target-domain evaluation shows that VideoDPO-bridge DPO
does not transfer or target-domain DPO data is required.

Exp9 first gate:

- data: D3 `selected_primary_comp.repaired.pai_paths.jsonl` if path rewrites
  were needed, otherwise `selected_primary_comp.repaired.jsonl`
- task: partial-mask training
- `train_mask_mode=partial`
- `mask_from_manifest=true`
- `loss_region_mode=full` first
- winner-anchored DPO
- beta = 10
- `winner_abs_reg_weight=0.05`
- `winner_gap_reg_weight=1.0`
- `lose_gap_weight=0.25` or no-lose gate
- Stage1 DPO only
- frozen SFT / target-domain Stage2
- no DPO Stage2
- 1000 or 1500 step gate before any long run

Prepared launchers:

```text
scripts/launch_exp9_youtubevos_d3_partialmask_wingap_stage1_gate_pai.sh
scripts/launch_exp9_youtubevos_d3_partialmask_wingap_nolose_stage1_gate_pai.sh
scripts/launch_exp9_youtubevos_d3_nocomp_partialmask_wingap_stage1_gate_h20.sh
```

The no-lose script is a fallback only and must not be launched unless explicitly
requested.

Comp-vs-nocomp target-domain plan:

- PAI: D3 selected-primary-comp Stage1 gate.
- H20: D3 selected-primary-nocomp Stage1 gate on GPUs 0-5.
- Compare both on YouTube-VOS / DAVIS with `inference/metrics.py` through the
  target-domain wrapper.
- Do not use VBench for this inpainting comparison.

Decision matrix:

- If comp improves and nocomp does not, use comp.
- If nocomp improves and comp does not, use nocomp.
- If both improve, compare metrics and qualitative temporal stability before a
  Stage1 sweep.
- If both fail, stop direct DPO and consider target-domain SFT warmup or a
  no-lose gate.

## 2026-06-03 Gate Monitor Update

PAI comp gate:

```text
experiment = exp9_youtubevos_d3_partialmask_wingap_lose025_stage1_gate1500
status = manually stopped after overshooting gate; invalid as Exp9 gate
stop_evidence = about 4856 / 10000
stop_report = reports/pai_exp9_comp_gate_stop_report.md
checkpoint_status = checkpoint-2000 and checkpoint-4000 exist under stale Exp5-named output dir
stale_output_dir = /mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260603_065327_exp5_d2_comp_k4_stage2_full
```

The run should not be resumed as a 10000-step long training and should not be
reported as a valid Exp9 gate. The launcher has been patched so future Exp9
gate runs ignore stale `RUN_NAME`, `MAX_STEPS`, `CKPT_STEPS`, `CKPT_LIMIT`,
`VAL_STEPS`, and `LOGGING_STEPS` unless an explicit config override flag is
set.

H20 nocomp gate:

```text
experiment = exp9_youtubevos_d3_nocomp_partialmask_wingap_lose025_stage1_gate1500_h20
status = finished normally
monitor_report = /home/nvme01/H20_Video_inpainting_DPO/reports/h20_exp9_nocomp_gate_monitor_report.md
manifest = /home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4/manifests/selected_primary_nocomp.jsonl
max_steps_detected = 1500
current_step = 1500 / 1500 at 2026-06-04 01:08 CST
checkpoint_status = checkpoint-500, checkpoint-1000, checkpoint-1500, last_weights
```

H20 nocomp is ready for YouTube-VOS / DAVIS inpainting evaluation through
`tools/run_inpainting_metric_eval.py` and `inference/metrics.py`.

## 2026-06-04 CST Eval Execution Update

PAI clean comp gate:

```text
status = running
launch_report = reports/pai_exp9_comp_clean_launch_report.md
verified = D3 selected-primary-comp PAI manifest, partial mask,
           mask_from_manifest=true, Stage1 only, max_steps=1500,
           checkpoint every 500
```

H20 nocomp target eval:

```text
status = completed for D3 selected-primary-nocomp / YouTube-VOS-derived rows
log = logs/pipelines/exp9_d3_nocomp_target_eval_20260604_023243.log
output = logs/target_eval/exp9_d3_nocomp_gate_h20_20260604_023243
baseline = DiffuEraser-base
candidate = Exp9 nocomp last_weights
```

Metric summary:

| model | mask PSNR | mask SSIM | boundary PSNR | outside mean diff | temporal delta vs GT |
| --- | ---: | ---: | ---: | ---: | ---: |
| DiffuEraser-base | 11.2407 | 0.2885 | 23.3338 | 3.0861 | 8.7626 |
| Exp9 nocomp last | 11.7119 | 0.2931 | 19.2961 | 3.1053 | 13.5626 |

Interpretation: nocomp helps the masked crop slightly but worsens boundary
PSNR, outside-region max/diff, and temporal stability. It is not enough to
select nocomp for a 3000-step sweep until clean PAI comp is evaluated.

Checkpoint-500, checkpoint-1000, and checkpoint-1500 are present but are
accelerator state directories (`model.safetensors`, optimizer/scheduler state).
They need conversion/export to `unet_main/` + `brushnet/` before direct
inference evaluation. `last_weights` is already exported and is directly
evaluable.

DAVIS remains blocked until target-domain prediction videos and a validated
pair manifest are produced. Do not substitute VBench for DAVIS inpainting
metrics.

## Do Not Do

- Do not start new Exp9 training beyond the current gates.
- Do not jump to Exp8.
- Do not continue VideoDPO warmup.
- Do not run full VBench.
- Do not regenerate D2.
- Do not delete or overwrite D3 H20 originals.
- Do not train from manifests containing `/home/nvme01/...` paths on PAI.
- Do not treat VideoDPO partial-mask eval as final result.
