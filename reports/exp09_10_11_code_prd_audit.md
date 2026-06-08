# Exp9 / Exp10 / Exp11 Code and PRD Audit

Date: 2026-06-08

## Scope Read

PRD files reviewed:

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/08_experiment_results_20260602.md`
- `PRD/10_target_domain_youtubevos_davis_plan.md`
- `PRD/dpo_diagnostics_and_metrics_plan.md`

Code reviewed:

- `training/dpo/train_stage1.py`
- `training/dpo/train_stage2.py`
- `training/dpo/scripts/03_dpo_stage1.sbatch`
- `training/dpo/scripts/03_dpo_stage2.sbatch`
- `training/dpo/scripts/04_dpo_stage2.sbatch`
- `training/dpo/scripts/run_stage1.py`
- `training/dpo/scripts/run_stage2.py`
- `training/dpo/dataset/generated_loser_manifest_dataset.py`
- `training/dpo/dataset/factory.py`
- `tools/run_inpainting_metric_eval.py`
- `inference/metrics.py`
- `inference/run_BR.py`
- `diffueraser/diffueraser.py`
- `scripts/launch_exp8_d3_comp_regionloss_s1s2_2000_davis_pai.sh`

## Findings Before Patch

1. `compute_dpo_loss` supported raw gaps and loser-gap weighting, but not `gap_normalization=log_ratio`, `gap_eps`, or `lose_gap_clip_tau`.
2. Stage1 and Stage2 exposed `beta_dpo`, `lose_gap_weight`, winner-side regularizers, and `loss_region_mode`, but did not expose normalized-gap or clipping arguments.
3. `LOSS_REGION_MODE=region` existed, but the weighted MSE used `mean(err * weight)` semantics. Exp10 requires `sum(weight * mse) / sum(weight)`.
4. `GeneratedLoserManifestDataset` requires `win_video_path`, `final_loser_video_path`, and `mask_path`; it uses `win_video_path` as the positive/winner clip.
5. The dataset returned `sample_id`, `pair_index`, and `manifest_row`, but Stage1/Stage2 collate functions did not preserve them for per-sample diagnostics.
6. DAVIS validation is reusable through `inference/run_BR.py` and the Exp8 DAVIS wrapper pattern.
7. `inference/run_BR.py` already uses ProPainter prior before DiffuEraser and produces per-sample `comparison_4in1.mp4`, but raw6/no-PCM/no-blur settings were not explicit CLI settings.
8. Metrics are routed through `tools/run_inpainting_metric_eval.py`, which delegates metric computation to `inference/metrics.py`. VBench is not needed for inpainting conclusions.
9. ProPainter/RAFT paths are runtime-dependent; the PAI wrapper must check and fail early if the RAFT file is corrupt.
10. Train-time differentiable flow/prior consistency for Exp11 is not safely available in the current Stage1/Stage2 loops.

## Implemented

1. Added `gap_normalization`, `gap_eps`, and `lose_gap_clip_tau` to Stage1/Stage2 DPO loss.
2. Implemented `log_ratio` normalized gaps:
   - `norm_win_gap = log((m_w + eps) / (m_w_ref + eps))`
   - `norm_lose_gap = log((m_l + eps) / (m_l_ref + eps))`
   - `norm_lose_gap_clipped = clamp(norm_lose_gap, max=lose_gap_clip_tau)`
3. Kept loser-gap in the DPO term; no-lose-gap remains diagnostic only.
4. Changed region-local weighted MSE to normalized weighted MSE.
5. Added region diagnostics:
   - `mask_region_mse`
   - `boundary_region_mse`
   - `outside_region_mse`
   - `mask_area_ratio`
   - `boundary_area_ratio`
   - `outside_area_ratio`
   - `region_weight_sum`
6. Added DPO gap diagnostics:
   - `raw_win_gap`
   - `raw_lose_gap`
   - `norm_win_gap`
   - `norm_lose_gap`
   - `norm_lose_gap_clipped`
   - `gap_normalization`
   - `gap_eps`
   - `lose_gap_clip_tau`
7. Added `dpo_gap_trace.csv` and `dpo_gap_samples.jsonl.gz` output paths for Stage1/Stage2.
8. Preserved `sample_id`, `pair_index`, and `mask_area_ratio` in Stage1/Stage2 batch collation for per-sample debug.
9. Exposed DAVIS eval settings in `inference/run_BR.py`:
   - `--num_inference_steps`
   - `--use_pcm`
   - `--apply_gaussian_blur`
   - `--hard_comp`
10. Added independent experiment folders and registry entries for Exp9, Exp10, and Exp11.
11. Added PAI master pipeline `scripts/launch_exp09_10_11_pai.sh`, defaulting to `RUN_EXPERIMENTS=exp9`.

## Exp11 Audit Result

Status: blocked.

The repository can run DAVIS inference with ProPainter prior, but the training loop does not yet expose a safe differentiable flow tensor or an image/x0-space ProPainter-prior consistency target. Therefore Exp11 must not launch by default. The pipeline writes `reports/exp11_flow_prior_implementation_audit.md` and stops before training unless this is resolved.

## Validation Contract

All Exp9/10/11 validation scripts use:

- DAVIS path: `/mnt/workspace/hj/nas_hj/data/external/davis_432_240`
- SFT-48000 DiffuEraser baseline: `/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000`
- ProPainter prior
- `NUM_INFERENCE_STEPS=6`
- `USE_PCM=false`
- `MASK_DILATION=0`
- `APPLY_GAUSSIAN_BLUR=false`
- `HARD_COMP=true`
- Metric backend: `inference/metrics.py`
- Metric wrapper: `tools/run_inpainting_metric_eval.py`
- VBench: not used

## Runtime Stop Conditions

The PAI wrapper stops before training if:

- the D3 manifest is missing,
- the manifest contains `/home/nvme01`,
- `win_video_path` is missing or appears to be a generated loser/candidate path,
- sampled manifest paths are missing,
- SFT-48000 weights are missing,
- DAVIS paths are missing,
- ProPainter/RAFT prior is missing or corrupt,
- py_compile or bash syntax checks fail,
- Exp11 is requested before its implementation audit passes.
