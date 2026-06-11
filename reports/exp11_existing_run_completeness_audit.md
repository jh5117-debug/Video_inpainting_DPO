# Exp11 Existing Run Completeness Audit

Date: 2026-06-11

Interpretation: this is `Exp11-proxy`, not real optical-flow / ProPainter-prior consistency DPO.

stage1_complete = true
stage2_complete = true

stage1_last_weights: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260609_2331_exp11_n16_gpus4_7_scratch_exp11_flow_prior_consistency_dpo_s1_2000_davis_pai/last_weights`
stage2_last_weights: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260609_2331_exp11_n16_gpus4_7_scratch_exp11_flow_prior_consistency_dpo_s2_2000_davis_pai/last_weights`
stage1_dpo_diag: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260609_2331_exp11_n16_gpus4_7_scratch_exp11_flow_prior_consistency_dpo_s1_2000_davis_pai/dpo_diagnostics.csv`
stage2_dpo_diag: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260609_2331_exp11_n16_gpus4_7_scratch_exp11_flow_prior_consistency_dpo_s2_2000_davis_pai/dpo_diagnostics.csv`
pipeline_log: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp09_10_11_pai_sync/logs/pipelines/exp11_flow_prior_consistency_dpo_s1s2_2000_davis_pai_20260609_2331_exp11_n16_gpus4_7_scratch_20260609_232933.log`

## Stage Details

| stage | complete | checkpoints | latest_checkpoint | diag_rows | run_manifest | fatal_log_hits | missing |
|---|---:|---:|---|---:|---|---:|---|
| stage1 | True | 4 | `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260609_2331_exp11_n16_gpus4_7_scratch_exp11_flow_prior_consistency_dpo_s1_2000_davis_pai/checkpoint-2000` | 201 | `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260609_2331_exp11_n16_gpus4_7_scratch_exp11_flow_prior_consistency_dpo_s1_2000_davis_pai/run_manifest.json` | 0 | none |
| stage2 | True | 4 | `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260609_2331_exp11_n16_gpus4_7_scratch_exp11_flow_prior_consistency_dpo_s2_2000_davis_pai/checkpoint-2000` | 201 | `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260609_2331_exp11_n16_gpus4_7_scratch_exp11_flow_prior_consistency_dpo_s2_2000_davis_pai/run_manifest.json` | 0 | none |

## Missing Items

none
