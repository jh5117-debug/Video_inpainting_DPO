# DPO-Diag Audit

Updated: 2026-06-04

This document repairs a process gap: DPO diagnostics are not optional notes.
Every DPO experiment must have a dpo-diag artifact or be marked artifact
incomplete.

Evidence inputs:

- `reports/experiment_artifact_audit/h20_diag_files.txt`
- `reports/experiment_artifact_audit/h20_all_candidate_artifacts.txt`
- `reports/experiment_artifact_audit/dpo_diag_file_preview.txt`
- Existing PRD summaries for Old Exp5 collapse and New Exp5/New Exp6 reruns.

## 1. Experiments With dpo-diag Found

| Experiment | dpo-diag status | Evidence |
|---|---|---|
| New Exp6 | found | H20 found multiple `dpo_diagnostics.csv` files, including `experiments/dpo/stage1/20260601_004753_exp6_d2_nocomp_k4_wingap_lose025_beta10_s1s2_4000_stage1/dpo_diagnostics.csv` and `experiments/dpo/stage2/20260601_004753_exp6_d2_nocomp_k4_wingap_lose025_beta10_s1s2_4000_stage2/dpo_diagnostics.csv`. |
| Exp9 H20 nocomp | found | `experiments/dpo/stage1/20260603_131758_exp9_youtubevos_d3_nocomp_partialmask_wingap_lose025_stage1_gate1500_h20_stage1/dpo_diagnostics.csv`. |
| Exp9 H20 comp no-lose | found | `experiments/dpo/stage1/20260604_080411_exp9_youtubevos_d3_comp_wingap_nolose_stage1_gate1000_h20_stage1/dpo_diagnostics.csv`. |
| Earlier baseline diagnostics | found | `PRD/assets/dpo_metric_analysis_20260505/*diagnostics*.csv` includes VideoDPO, DiffDPO, and no-lose-gap diagnostic references. |

## 2. Experiments With Metrics But Missing or Unconfirmed dpo-diag

| Experiment | Metrics found | dpo-diag gap |
|---|---|---|
| Exp9 H20 nocomp target eval | `logs/target_eval/exp9_d3_nocomp_gate_h20_20260604_023243/metrics/summary.csv` | Training dpo-diag exists, but target eval metrics are not a replacement for dpo-diag. |
| Exp7 partial-mask / hybrid visual folders | Local MP4 samples exist in `/home/hj/dpo-2-1-exp`; prior partial-mask metric results were discussed. | Training dpo-diag was not found in the current H20 tree; likely PAI/NAS-side artifact needs manual search. |
| Old Exp5 / New Exp5 visual folders | Local MP4 samples exist in `/home/hj/dpo-2-1-exp/exp5` and `/home/hj/dpo-2-1-exp/new-exp5`. | Complete run folder and dpo-diag were not found in current H20 scan; PAI search required. |

## 3. Experiments With dpo-diag Missing

| Experiment | Missing status |
|---|---|
| Exp4 full-mask generated loser | No dpo-diag found; only local PNG evidence folder found. Treat as smoke / negative data-quality artifact unless PAI returns a full run. |
| Old Exp5 complete run | PRD records dpo-diag interpretation, but the CSV artifact was not found in current H20 scan. |
| New Exp5 complete run | No H20 dpo-diag artifact found in scan. |
| Exp7 complete training dpo-diag | Not found in current H20 scan. |
| Exp8 region loss | Not found; do not present as completed. |
| PAI Exp9 clean comp | Not found in H20 scan by definition; requires PAI manual search. |

## 4. Required dpo-diag Fields

Each DPO experiment diagnostic summary should include at least:

- checkpoint step / `global_step`
- `dpo_loss`
- `total_loss` or `anchored_total_loss`
- `implicit_acc`
- winner quality: `mse_w`, `ref_mse_w`, `win_gap`, `mse_w_over_ref_mse_w`
- loser gap: `mse_l`, `ref_mse_l`, `lose_gap`, `mse_l_over_ref_mse_l`
- preference gap / reward gap: `reward_margin`, `inside_term_mean`, `inside_term_min`, `inside_term_max`
- saturation signals: `sigma_term`, `kl_divergence`, `loser_dominant_ratio`
- regularization weights: `beta_dpo`, `winner_abs_reg_weight`, `winner_gap_reg_weight`, `winner_gap_reg_margin`, `lose_gap_weight`
- eval metrics linkage: VBench for video generation/full-mask bridge; `inference/metrics.py` wrapper for partial-mask inpainting.

The H20 preview confirms the modern winner-anchored CSV header includes:

```text
global_step,dpo_loss,implicit_acc,win_gap,lose_gap,mse_w,ref_mse_w,mse_l,
ref_mse_l,loser_dominant_ratio,sigma_term,grad_norm,total_loss,
anchored_total_loss,sft_loss,sft_reg_weight,lose_gap_weight,winner_abs_reg,
winner_abs_reg_weight,winner_gap_reg,winner_gap_reg_weight,
winner_gap_reg_margin,relu_win_gap_mean,relu_win_gap_max,
win_gap_positive_ratio,mse_w_over_ref_mse_w,mse_l_over_ref_mse_l,
reward_margin,kl_divergence,inside_term_mean,inside_term_min,inside_term_max
```

## 5. Old Exp5 Collapse Evidence

Existing PRD records Old Exp5 as the canonical collapse case:

- `implicit_acc` saturated near 1 while `dpo_loss` approached 0.
- `mse_w >> ref_mse_w`: the model damaged the winner.
- `mse_l >> ref_mse_l`: the model damaged the loser even more.
- DPO ranking looked correct because loser degradation was larger than winner degradation.
- Qualitative outputs showed high-frequency stripe/color collapse.

The important lesson is: **ranking success is not visual success**.

## 6. New Exp5 / New Exp6 Improvement Evidence

New Exp5 and New Exp6 introduced winner anchoring:

```text
beta_dpo = 10
winner_abs_reg_weight = 0.05
winner_gap_reg_weight = 1.0
winner_gap_reg_margin = 0.0
lose_gap_weight = 0.25
sft_reg_weight = 0.0
```

The dpo-diag columns needed to support the improvement claim are
`winner_gap_reg`, `mse_w_over_ref_mse_w`, `mse_l_over_ref_mse_l`,
`loser_dominant_ratio`, and `inside_term_mean`.

H20 confirms New Exp6 has dpo-diag files. New Exp5 still needs PAI artifact
search; without the CSV, PPT should say "qualitative improvement observed,
diag artifact pending" rather than making a complete quantitative claim.

## 7. Exp9 ckpt500 Early-Window Evidence

Exp9 target-domain comp eval previously showed:

- DiffuEraser-base: mask PSNR 11.139, mask SSIM 0.285.
- Exp9 comp ckpt500: mask PSNR 13.607, mask SSIM 0.364.
- Exp9 comp last: mask PSNR 12.373, mask SSIM 0.286.

This supports the early-window conclusion: ckpt500 is useful, later checkpoints
lose the advantage. The dpo-diag table must be used to determine whether later
degradation corresponds to loser-dominant shortcut, winner drift, or both.

PAI clean comp dpo-diag is still missing from the current H20 scan and must be
imported from PAI manual search.

## 8. dpo-diag Tables Needed For PPT

PPT should include small summary tables, not raw logs:

1. Old Exp5 collapse: `implicit_acc`, `dpo_loss`, `mse_w/ref`, `mse_l/ref`,
   visual verdict.
2. New Exp5 vs Old Exp5: winner anchoring enabled, visual stability improved,
   dpo-diag artifact pending if PAI CSV is unavailable.
3. New Exp6: no-comp diagnostic, H20 dpo-diag available.
4. Exp7: Stage1/Stage2 task-alignment gate; dpo-diag artifact needs PAI/NAS
   registry confirmation.
5. Exp9: ckpt500 vs later checkpoints; connect metric early-window result with
   dpo-diag once PAI clean comp CSV is returned.

