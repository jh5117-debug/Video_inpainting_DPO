# Exp26 VideoPainter Shadow-Dev Confirmatory Readback

- branch: `research/exp26-videopainter-dpo-v2`
- HEAD: `6c3160a5e4d542388d2594ec38787ab2fc8cb833`
- status: `runtime_snapshot_no_git`
- training_manifest_sha256: `82f661f2f30a581a213972533817624217eabb97eba7aaeedc00ee2109e4e716`
- search_dev_sha256: `41c6571d26e4a5130818dd50fbbe1314c9d953284561a3cd20f630572f7c2a71`
- shadow_dev_sha256: `0338dba1513cfe0e5dd85cbf793b3782902b981ac9610b0e472c6a048f738c02`
- run_root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/shadowdev_confirmatory_20260625`
- train_root: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp26_videopainter_dpo_v2/vp_primary32_50step_20260625_171032`
- primary_comparison: `Step50 vs fixed Step0 official initialization`
- secondary_trajectory: `Step10 and Step30 explanatory only`
- banned_repeats: no training, no data reselection, no search-dev retuning, no left CLI modification
- allowed_right_gpus: dynamic eligible subset of GPU0/GPU5/GPU6/GPU7 only

## Files Read

- `PRD/00_current_status.md`: exists
- `PRD/01_experiment_matrix.md`: exists
- `PRD/48_exp26_videopainter_dpo_v2.md`: exists
- `experiment_registry/exp26_videopainter_dpo_v2/status.md`: exists
- `experiment_registry/exp26_videopainter_dpo_v2/paths.yaml`: exists
- `experiment_registry/exp26_videopainter_dpo_v2/config.yaml`: exists
- `experiment_registry/exp26_videopainter_dpo_v2/metric_summary.md`: exists
- `experiment_registry/exp26_videopainter_dpo_v2/qualitative_summary.md`: exists
- `reports/exp26_gate64_manifest_identity.json`: exists
- `reports/exp26_vp_step0_baseline.md`: exists
- `reports/exp26_vp_l0_l1.md`: exists
- `reports/exp26_vp_10step.md`: exists
- `reports/exp26_vp_50step_final.md`: exists
- `reports/exp26_vp_50step_metrics.csv`: exists
- `reports/exp26_vp_50step_statistics.json`: exists
- `reports/exp26_vp_50step_diagnostics.csv`: exists
- `reports/exp26_vp_50step_visual_review.csv`: exists
- `exp26_videopainter_dpo_v2/code/run_vp2_gate64_official_generation.py`: exists
- `exp26_videopainter_dpo_v2/code/evaluate_vp2_step0_searchdev.py`: exists
- `exp26_videopainter_dpo_v2/code/materialize_vp2_49f_sources.py`: exists
- `exp26_videopainter_dpo_v2/code/generate_vp2_moving_br_masks.py`: exists
- `exp26_videopainter_dpo_v2/code/review_gate64_official_outputs.py`: exists

## Step Identity Targets

- Step0: `/mnt/nas/hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter/ckpt/VideoPainter/VideoPainter/checkpoints/branch`
- Step10: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp26_videopainter_dpo_v2/vp_primary32_50step_20260625_171032/checkpoint-10`
- Step30: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp26_videopainter_dpo_v2/vp_primary32_50step_20260625_171032/checkpoint-30`
- Step50: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp26_videopainter_dpo_v2/vp_primary32_50step_20260625_171032/checkpoint-50`

Shadow-dev has not been used for checkpoint selection by this right-side workflow.
