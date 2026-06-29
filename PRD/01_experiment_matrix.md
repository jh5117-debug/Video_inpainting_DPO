## 2026-06-24 Exp26 Gate16 Final Video Review

| User-facing name | Status | Evidence / registry |
| --- | --- | --- |
| Exp26 VideoPainter v2 Gate16 | `GATE16_PASSED_WITH_REJECTION`; Gate64 allowed as next milestone only | `reports/exp26_gate16_final_video_review.md`, `reports/exp26_gate16_final_video_review.csv`, `experiment_registry/exp26_videopainter_dpo_v2/status.md` |

Notes:

- Existing Gate16 rows only; no replacement of the failed row.
- Final buckets: `medium-hard=10`, `hard-plausible=5`, `trivial-bad=1`,
  `technical-invalid=0`.
- `vp2_gate16_BLENDER_CON001_00742` remains a true model failure.
- No Gate64 or DPO training was launched by this milestone.

## 2026-06-19 Exp20/21/22 Autoresearch Setup

| User-facing name | Status | Evidence / registry |
| --- | --- | --- |
| Exp20 scale-adaptive region-balanced DPO | precheck implemented; no training result yet | `experiment_registry/exp20_autoresearch_scale_adaptive_region_dpo`, `PRD/42_exp20_autoresearch_scale_adaptive_region_dpo.md` |
| Exp21 multibackbone VideoDPO BR smoke | compatibility matrix scaffold ready; real smoke pending | `experiment_registry/exp21_multibackbone_videodpo_br_smoke`, `PRD/43_exp21_multibackbone_videodpo_br_smoke.md` |
| Exp22 multimodel BR benchmark prep | asset scanner scaffold ready; real inference smoke pending | `experiment_registry/exp22_multimodel_br_benchmark_prep`, `PRD/44_exp22_multimodel_br_benchmark_prep.md` |

Current best remains `Exp11 boundary outer b0.75 S2`. Exp20 heavy PAI search is
not launched until legacy parity, locked dev split, and recomputed SFT/Exp11 dev
baselines pass.

## 2026-06-18 Exp19b Exploratory 2000

| User-facing name | Status | Evidence / registry |
| --- | --- | --- |
| Exp19b exploratory 2000 | completed DAVIS50; no-op / negative ablation; do not continue | `experiment_registry/exp19b_exploratory_2000`, `reports/exp19b_exploratory_2000_davis50_result.md`, `reports/exp19b_exploratory_2000_dpo_diag_summary.md` |

This run was intentionally launched after the earlier negative gate as an
exploratory check requested by the user.

DAVIS50 result:

| Method | PSNR | SSIM | LPIPS | Ewarp | strict mask PSNR | boundary PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exp11 boundary outer b0.75 S2 | 32.840213 | 0.971818 | 0.015339 | 7.181782 | 21.196763 | 26.441316 |
| Exp19b exploratory 2000 | 32.840122 | 0.971818 | 0.015340 | 7.181850 | 21.196671 | 26.441224 |

Decision:

```text
Current best remains Exp11 boundary outer b0.75 S2.
Do not continue Exp19b under this exact setup.
```

## 2026-06-18 Exp19 Flow-Adapter Gate

| User-facing name | Status | Evidence / registry |
| --- | --- | --- |
| Exp19b boundary-gated Stage2 flow adapter 500 | DAVIS10 eval completed; negative/neutral gate; do not expand | `experiment_registry/exp19_boundary_gated_flow_adapter_dpo`, `reports/exp19_final_report.md`, `reports/exp19b_davis10_metric_summary.md`, `reports/exp19b_visual_case_judgement.md` |

Summary:

- isolated Stage2 hook wrapper implemented under `exp19_boundary_gated_flow_adapter_dpo/`.
- unsafe `additional_residuals` interfaces are no longer used.
- flow cache limit100 completed with ProPainter completed bidirectional flow and forward-backward confidence.
- zero-init / gradient preflight passed.
- Exp19b adapter-only 500-step gate completed and saved `checkpoint-250`, `checkpoint-500`, and `last_weights`.
- Exp19 inference wrapper was implemented and strict-loaded `flow_adapter.pt`.
- DAVIS10 metric/visual gate completed with real flow-adapter inference.

DAVIS10 result:

| Method | PSNR | SSIM | LPIPS | Ewarp | strict mask PSNR | boundary PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exp11 boundary outer b0.75 S2 | 29.8295 | 0.9633 | 0.02065 | 8.3307 | 18.5317 | 24.6577 |
| Exp19b Stage2-500 | 29.8291 | 0.9633 | 0.02065 | 8.3306 | 18.5313 | 24.6574 |

Decision:

```text
Do not expand Exp19 to 1000 / DAVIS50 / full-cache.
```

Exp19b is visually safe but effectively tied with Exp11 and fails the temporal
positive gate. Current best remains Exp11 boundary outer b0.75 S2.

## 2026-06-18 Exp19-R0 / Exp19c Flow Refinement

| User-facing name | Status | Evidence / registry |
| --- | --- | --- |
| Exp19-R0 flow adapter calibration | complete; parity fixed; causality tiny-positive only | `experiment_registry/exp19r0_flow_adapter_calibration`, `reports/exp19_inference_parity_repair.md`, `reports/exp19_residual_scale_confidence_sweep.md`, `reports/exp19r0_flow_causality_audit.md` |
| Exp19c light latent-warp DPO | complete negative ablation; do not expand | `experiment_registry/exp19c_light_warp_dpo`, `reports/exp19c_davis10_metric_summary.md`, `reports/exp19c_visual_case_judgement.md`, `reports/exp19_refinement_final_report.md` |
| Exp19d motion-aware flow DPO | gated off; not launched | `experiment_registry/exp19d_motion_aware_flow_dpo` |

Decision:

```text
Exp19c fails the positive gate. Do not run Exp19d, DAVIS50, full cache, or 1000/2000-step continuation.
```

## 2026-06-15 Exp11/Exp12 Boundary-Normalization Batch

| User-facing name | Status | Evidence / registry |
| --- | --- | --- |
| Exp11 boundary inner b0.75 | complete | `experiment_registry/exp11_region_boundary_ablation` |
| Exp11 boundary outer b0.75 S1 | complete | `experiment_registry/exp11_region_boundary_ablation` |
| Exp11 boundary outer b0.75 S2 | **current best** | `experiment_registry/exp11_region_boundary_ablation`, `PRD/22_this_week_exp11_outer_b075_s2_best_and_visual_cases.md` |
| Exp11 boundary both b0.75 | complete | `experiment_registry/exp11_region_boundary_ablation` |
| Exp11 boundary both b1.0 | complete | `experiment_registry/exp11_region_boundary_ablation` |
| Exp12 adaptive norm | complete negative/ablation | `experiment_registry/exp12_adaptive_normalization` |
| Exp12 adaptive + outer b0.75 | complete negative/ablation | `experiment_registry/exp12_adaptive_outer_boundary` |
| YouTubeVOS100 extension | running / collecting evidence | `PRD/23_youtubevos100_davis50_extended_eval.md` |
| Adapter feasibility | complete, no training launched | `PRD/24_dpo_adapter_baseline_feasibility.md` |

Fixed metric protocol for all current claims:

```text
DAVIS50 / YouTubeVOS100 raw6 hard-comp, D+G off, no PCM, frame-wise in-memory metrics.
```

Current best:

```text
Exp11 boundary outer b0.75 S2, PSNR=33.013954, SSIM=0.972295, LPIPS=0.015363, VFID=0.175423, TC=0.971122.
```

## 2026-06-09 Active Experiment Ledger

The active experiment list has been compacted. Legacy gates that are no longer
part of the main structure are kept under `pending_delete/` rather than deleted.

| User-facing name | Status | Evidence / registry |
| --- | --- | --- |
| pre-Exp5 historical setup | historical context | `experiment_registry/exp01_*` to `exp04_*` |
| Exp5 | failed diagnostics | `experiment_registry/exp05_old_d2_comp_plain_failed`, `experiment_registry/exp05_beta10_plain_failed` |
| NewExp5 | completed, improved but not final | `experiment_registry/exp05_new_d2_comp_wingap_lose025` |
| NewExp6 | completed diagnostic | `experiment_registry/exp06_new_d2_nocomp_wingap_lose025` |
| Exp7a-1 | completed/evaluated artifact family | `experiment_registry/exp07_fix_smallmask_prior` / Exp7a evidence |
| Exp7a-2 | completed/evaluated artifact family | `experiment_registry/exp07_fix_smallmask_prior` / Exp7a evidence |
| Exp8a-1 | completed negative | `experiment_registry/exp08a_d3_comp_fullloss_davis_s1s2_2000` |
| Exp8a-2 | completed negative | `experiment_registry/exp08a_d3_comp_fullloss_davis_s1s2_2000` |
| Exp8c-1 | completed/diagnostic | `experiment_registry/exp08c_youtubevos_gtwin_d3comp_fullloss_davis_s1s2_2000` |
| Exp8c-2 | completed/diagnostic | `experiment_registry/exp08c_youtubevos_gtwin_d3comp_fullloss_davis_s1s2_2000` |
| Exp9-1 | complete | `experiment_registry/exp09_logratio_gap_dpo` Stage1 DPO + SFT Stage2 DAVIS |
| Exp9-2 | complete | `experiment_registry/exp09_logratio_gap_dpo` Stage1 DPO + Stage2 DPO DAVIS |
| Exp10-1 | partially complete; fresh no-resume retry also SIGTERM-killed | `experiment_registry/exp10_region_local_dpo` |
| Exp11 | blocked | `experiment_registry/exp11_flow_prior_consistency_dpo`; audit must pass before training |

Exp10 retry rule: do not resume the interrupted PAI checkpoint unless explicitly
debugging checkpoint loading. Use a fresh run:

```bash
bash scripts/pai_launch_exp10_fresh_gpus0_6.sh
```

Exp11 rule: no fake training. If train-time flow/prior consistency is not
implemented and audited, Exp11 remains blocked.

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

## 2026-06-05 Exp8 DAVIS S1/S2 Region-Loss Row

| Experiment | Status | Data | Task | Loss | Eval | Next |
| --- | --- | --- | --- | --- | --- | --- |
| Exp8 D3 comp region-loss DAVIS S1/S2 2000 | prepared for PAI manual launch | D3 selected-primary-comp repaired PAI manifest | partial-mask video inpainting | `L = -logσ{-0.5*10*(win_gap - 0.25*lose_gap)} + 0.05*m_w + ReLU(win_gap)`, with `m_w/m_l` as region-weighted MSE and weights mask=1.0, boundary=0.5, outside=0.05 | DAVIS four-column visualization and `tools/run_inpainting_metric_eval.py`; no VBench | Run only after PAI precheck passes; H20 remains untouched while data generation runs |

# Experiment Matrix

## 2026-06-29 Exp40 MiniMax PSNR-Safe Rescue

| User-facing name | Status | Evidence / registry |
| --- | --- | --- |
| Exp40 LocalDPO v3 pool | `MINIMAX_LOCALDPO_V3_POOL_READY_MINIMUM`; ready for Step0/SFT diagnostics only | `PRD/55_exp40_minimax_psnr_safe_rescue.md`, `experiment_registry/exp40_minimax_psnr_safe_rescue`, `reports/exp40_localdpo_v3_pool.md`, `reports/exp40_localdpo_v3_summary.json` |

Notes:

- VOR-Train only; VOR-Eval not used.
- Selected pool is `train64/search24/shadow24`, not the full target
  `train96/search32/shadow32`.
- No MiniMax quality-positive claim exists yet.
- GPU2-GPU7 remain untouched by Exp40.

## 2026-06-18 Exp19 Flow-Adapter Attempt

| User-facing name | Status | Evidence / registry |
| --- | --- | --- |
| Exp19 Boundary-Gated Flow-Adapter DPO | superseded by completed Exp19b gate above | `exp19_boundary_gated_flow_adapter_dpo/`, `experiment_registry/exp19_boundary_gated_flow_adapter_dpo/`, `PRD/40_exp19_boundary_gated_flow_adapter_dpo.md`, `reports/exp19_final_report.md` |

This earlier architecture blocker is now superseded. Exp19b trained and
evaluated, but the completed DAVIS10 gate did not beat Exp11.

## 2026-06-18 Exp18 PAI Gate Result

| User-facing name | Status | Evidence / registry |
| --- | --- | --- |
| Exp18 Multi-frame Propagation-Confidence Gated DPO | Stage1-500 PAI gates complete; negative/exploratory ablation; do not expand | `exp18_multiframe_propagation_gated_dpo/`, `experiment_registry/exp18_multiframe_propagation_gated_dpo/`, `PRD/39_exp18_multiframe_propagation_gated_dpo.md`, `reports/exp18_final_pai_gate_report.md`, `reports/exp18_davis10_metric_summary.md`, `reports/exp18_visual_case_judgement.md` |

Exp18 completed the required first gate:

```text
limit=100 propagation cache
Exp18a Stage1-500
Exp18b Stage1-500
Exp18c oracle Stage1-500 diagnostic
DAVIS10 visual/metric sanity
```

DAVIS10 result:

| Method | PSNR | SSIM | strict mask PSNR | boundary PSNR |
| --- | ---: | ---: | ---: | ---: |
| Exp11 boundary outer b0.75 S2 | 30.2413 | 0.9650 | 18.7114 | 24.8326 |
| Exp18a prop-only S1-500 | 30.1024 | 0.9650 | 18.5725 | 24.7090 |
| Exp18b prop+gen S1-500 | 29.6892 | 0.9609 | 18.1593 | 24.7152 |
| Exp18c oracle S1-500 | 29.7626 | 0.9632 | 18.2326 | 24.7991 |
| SFT-48000 baseline | 30.0126 | 0.9635 | 18.4827 | 24.4772 |

Decision:

```text
No Exp18 variant beats Exp11. Do not run Stage1 1000, full cache, Stage1 2000, or Stage2.
Current best remains Exp11 boundary outer b0.75 S2.
```

## 2026-06-17 Exp16 Mainline Candidate

| User-facing name | Status | Evidence / registry |
| --- | --- | --- |
| Exp16 Prior-Confidence Gated DPO | Stage1 500 limit=100 + DAVIS10 sanity completed; weak positive signal vs SFT, not better than Exp11; do not full-train yet | `exp16_prior_confidence_gated_dpo/`, `experiment_registry/exp16_prior_confidence_gated_dpo/`, `PRD/36_exp16_prior_confidence_gated_dpo.md`, `reports/exp16_stage1_500_visual_case_judgement.md`, `reports/exp16_stage1_500_davis10_metric_summary.md` |
| Exp17 Saturation-Aware Positive DPO | Stage1 1000 gates completed; negative ablation; best variant Exp17b still below Exp11, so no Stage1 2000 / Stage2 | `exp17_saturation_positive_dpo/`, `experiment_registry/exp17_saturation_positive_dpo/`, `PRD/37_exp17_saturation_positive_dpo.md`, `reports/exp17_davis10_gate_metric_summary.md`, `reports/exp17_visual_case_judgement.md` |

Exp16 inherits the current best setting from Exp11 boundary outer b0.75 S2, but
adds prior-confidence gated latent-x0 losses. The limit=100 prior cache,
preflight, Stage1 500 small gate, confidence diagnostic fix, and DAVIS10 sanity
eval completed on PAI. It must not be reported as a completed result: DAVIS10
shows weak positive signal over SFT-48000, but Exp16 does not beat Exp11 outer
b0.75 S2 and negative cases remain.

Current Exp16 decision:

```text
Do not run full prior cache / Stage1 2000 / Stage2 2000 yet.
```

Paused while Exp16 is explored:

- OR benchmark
- BR / VideoPainter adapter
- adaptive normalization new variants
- additional Exp11 / Exp12 tuning

Exp17 was run as the next overnight gate and completed as a negative ablation.
Exp16 full prior cache and full training remain paused. Current best remains
Exp11 boundary outer b0.75 S2.

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


## 2026-05-31 Exp5 Collapse And Winner-Anchored Rerun

Old `exp5_d2_comp_k4_stage1/stage2_full` with `beta_dpo=500` and 10000-step
Stage1/Stage2 is now marked **failed / collapsed / diagnostic only**.

This failure is not a task failure: Exp3 showed the VideoDPO-to-DiffuEraser DPO
bridge can work. The old Exp5 failure is an optimization/preference-data
failure caused by D2 generated losers plus full-mask training, full-video loss,
`beta_dpo=500`, no SFT regularization, and long training. Early `acc=1`,
`dpo=0`, and `loss=0` are treated as DPO saturation, not as visual success.

The replacement reruns are:

| Experiment | Status | Manifest | beta_dpo | Stage1 steps | Stage2 steps | Validation during training | Post Stage2 eval |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `exp5_d2_comp_k4_wingap_lose025_beta10_s1s2_4000` | planned/running | `selected_primary_comp.repaired.jsonl` | 10 | 4000 | 4000 | disabled | qual30 + full VBench |
| `exp6_d2_nocomp_k4_wingap_lose025_beta10_s1s2_4000` | planned/running | `selected_primary_nocomp.repaired.jsonl` | 10 | 4000 | 4000 | disabled | qual30 + full VBench |

The intermediate `exp5_d2_comp_k4_beta10_s1s2_4000` rerun is also marked
**failed / collapsed / diagnostic only**. Stage2 loaded Stage1 correctly, but
diagnostics showed `mse_w >> ref_mse_w`, `mse_l >> ref_mse_l`, near-saturated
`implicit_acc`, and low DPO loss while qualitative outputs became universal
stripe/high-frequency textures. This confirms the problem is the unanchored DPO
objective, not a launcher or evaluation bug.

Old H20 Exp6 unanchored training is superseded by the winner-anchored rerun and
should be stopped if still running.

Winner-anchored rerun parameters:

```text
beta_dpo = 10
winner_abs_reg_weight = 0.05
winner_gap_reg_weight = 1.0
winner_gap_reg_margin = 0.0
lose_gap_weight = 0.25
sft_reg_weight = 0.0
stage1_steps = 4000
stage2_steps = 4000
```

## 2026-05-31 Exp7 Partial-Mask Task Gate

Exp5 winner-anchored improved the failure mode but is not final: the winner
anchor held `win_gap` down, yet qualitative outputs still show texture/color
attractors under the data-only full-mask/full-video bridge. Exp6 no-comp is
running on H20 and must continue for the comp-vs-no-comp comparison.

Exp7 is the next gate and is a **task ablation**, not a data-only run. It keeps
the D2 comp manifest but changes the training task so DiffuEraser sees the same
partial mask used during loser generation.

| Experiment | Status | Manifest | Train mask | Mask source | beta_dpo | lose_gap_weight | winner_abs_reg | winner_gap_reg | Stage1 | Stage2 | Gate validation |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `exp7_d2_comp_k4_partial_wingap_lose025_beta10_s1s2_gate1500` | planned / launching on PAI | `selected_primary_comp.repaired.jsonl` | partial | manifest `mask_path` | 10 | 0.25 | 0.05 | 1.0 | 1500 | 1500 | qual30 SBS + DPO diag summary |

Exp7 interpretation rule:

- If Exp7 gate is more stable than Exp5, D2 generated losers are not the sole
  problem; the data-only full-mask objective was mismatched with the partial
  mask generation process.
- If Exp7 gate still collapses, the next change should be data/prompt quality
  or a stronger winner-preservation strategy, not a direct full 4000+4000 run.

Exp7 gate1500 full-mask qual30 result update:

| Experiment | Current status | Observed eval | Interpretation | Next action |
| --- | --- | --- | --- | --- |
| `exp7_d2_comp_k4_partial_wingap_lose025_beta10_s1s2_gate1500` | inconclusive / risky | full-mask qual30 failed; stripe-heavy; some samples worse than new Exp5 | task-mismatched eval because training is partial-mask inpainting | run true partial-mask manifest eval before deciding failure |

Observed diagnostics:

- `winner_gap_reg_weight=1.0` keeps `win_gap` relatively bounded.
- `loser_dominant_ratio` can reach 1.0 for late steps.
- `mse_l_over_ref_mse_l` can become very high, so loser degradation remains a
  strong shortcut even with winner anchoring.

Exp7-PM-Gate1500 true partial-mask eval update:

```text
eval_name = Exp7-PM-Gate1500
output = /mnt/nas/hj/H20_Video_inpainting_DPO/logs/partialmask_eval/exp7_gate1500_20260602_000500
metric_samples = 100
qualitative_side_by_side = 60 videos
base = DiffuEraser-base converted_weights_step48000
evaluated_exp7 = Stage1_last, Stage2_last
skipped = Stage1_ckpt500 and Stage1_ckpt1000 because exported paths were absent
```

| Model | mask-region PSNR | mask-region SSIM | Status |
| --- | ---: | ---: | --- |
| DiffuEraser-base | 8.99765 | 0.272146 | baseline |
| Exp7 Stage1_last | 9.57079 | 0.288404 | best evaluated checkpoint; beats base on true partial-mask metrics |
| Exp7 Stage2_last | 7.88448 | 0.235938 | regressed below base and Stage1 |

Decision:

- Exp7 full-mask qual30 remains failed / task-mismatched.
- Exp7 partial-mask task alignment is promising because Stage1_last beats
  DiffuEraser-base on the task-matched metric gate.
- Stage2 appears harmful in this configuration, consistent with the
  loser-degradation shortcut seen in diagnostics.
- Do not launch Exp7 full 4000+4000 yet.
- Review the 60 partial-mask side-by-side videos, then prefer a Stage1-focused
  or no-lose-gap follow-up over a direct longer Stage2 run.

Prepared but not launched:

| Experiment | Status | Purpose |
| --- | --- | --- |
| `exp7_d2_comp_k4_partial_wingap_nolose_beta10_s1s2_gate1000` | script prepared only | cut `lose_gap_weight` to 0.0 if partial-mask eval confirms loser-degradation shortcut |

## 2026-06-02 DPO-S1 + SFT-S2 Hybrid Correction

The Exp7-PM-Gate1500 checkpoint comparison requires a stage-aware
interpretation. DiffuEraser Stage1 is the spatial/appearance stage; Stage2 is
the temporal/motion stage. Because Stage1_last beats DiffuEraser-base on true
partial-mask metrics while Stage2_last regresses, the next candidate is not
"Stage1-only inference." The correct candidate is:

```text
DPO Stage1 spatial/BrushNet/UNet2D weights
+
frozen SFT Stage2 temporal/motion weights
```

Do not train DPO Stage2 in the next step. Do not load a full SFT Stage2
checkpoint in a way that overwrites DPO Stage1 spatial weights.

New matrix rows:

| Experiment | Status | Stage1 source | Stage2 source | Task | Eval | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `exp7_pm_dpoS1_sftS2_hybrid_ckptsweep` | script prepared / audit first | Exp7 DPO Stage1 checkpoints and `last_weights` | best found SFT Stage2, then previous SFT, then official/base Stage2 | true partial-mask inpainting | partial-mask metrics + side-by-side only | First priority; no training and no full VBench. |
| `exp7_pm_stage1only_ckptsweep_wingap_lose025_beta10` | launcher prepared only | DPO Stage1 train 3000 with ckpt every 500 | none during training | partial-mask DPO | no automatic eval | Prepared only to produce better DPO-S1 candidates; do not launch until hybrid audit says more Stage1 checkpoints are needed. |

Hybrid evaluation must answer whether:

- DPO Stage1 spatial weights are preserved.
- SFT Stage2 motion weights are preserved.
- YouTube-VOS SFT Stage2 exists and should be used.
- The hybrid beats DiffuEraser-base.
- The hybrid beats Exp7 DPO Stage1 + DPO Stage2.
- Stage2 DPO should remain stopped.

## Core Ablation Directions This Week

The core plan has four directions, with Direction 2 split into 2A/2B:

1. `official_videodpo_diffueraser_data_fullmask_loser`
2. `official_videodpo_diffueraser_data_partialmask_loser_comp_k4`
3. `official_videodpo_diffueraser_data_partialmask_loser_nocomp_k4`
4. `official_videodpo_diffueraser_task_partialmask`
5. `official_videodpo_diffueraser_youtubevos_partialmask_data`

The important boundary:

- Experiments 1, 2A, and 2B are **data-only ablations**. Masks are used only to generate offline losers. Training still uses the completed `official_videodpo_diffueraser` full-mask bridge, so the training model does not receive partial masks.
- Experiment 3 is the first **task ablation**. The partial mask becomes a training-time model input, so DiffuEraser actually performs partial video inpainting.
- Experiment 4 is a **data-source ablation** on top of the Experiment 3 partial-mask task setting, moving from VideoDPO data to YouTube-VOS-derived data.

## Priorities

| Priority | Work | Notes |
| --- | --- | --- |
| 0 | Protect completed experiments | Do not break `official_videodpo_vc2`, `official_videodpo_diffueraser`, or DiffuEraser reproduction/SFT scripts. |
| 1 | PAI audit | Confirm model envs/weights, data paths, and completed output artifacts. |
| 2 | Full-mask loser data ablation | Data-only; win is VideoDPO winner, loser is full-mask inpainting output. |
| 3 | Partial-mask offline + comp | First priority for partial masks; cleanest data-only ablation. |
| 4 | Partial-mask offline + no-comp | Second priority; diagnostic comp-vs-no-comp ablation. |
| 5 | Partial-mask training task | Task ablation; partial mask becomes training condition. |
| 6 | YouTube-VOS partial-mask data | Data source ablation built on partial-mask task setting. |
| Future | Online loser generation | Not first version; document only until offline generation is stable. |

## Matrix

| Experiment | Status | Model | Data source | Win source | Loser source | Mask for loser generation | Mask for training | Comp | Offline/Online | Changed variable | Metrics | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `diffueraser_reproduction_sft` | completed | DiffuEraser | DAVIS / YouTube-VOS-derived | source video | reconstruction/inpainting output | task-specific | task-specific | setting-dependent | offline | reproduction/SFT/metric setting | PSNR, SSIM, VBench | Best eval: 6 steps, no PCM, no Gaussian blur, frame-wise metric transfer. |
| `official_videodpo_vc2` | completed | VC2 | VideoDPO | VideoDPO winner | VideoDPO rejected | none | none | false | existing pairs | official baseline | VBench, SBS | Completed full VBench. |
| `official_videodpo_diffueraser` | completed | DiffuEraser | VideoDPO | VideoDPO winner | VideoDPO rejected | none | full | false | existing pairs | model adapter | VBench, SBS, DPO diagnostics | Official VideoDPO skeleton + DiffuEraser full-mask bridge. |
| `official_videodpo_diffueraser_data_fullmask_loser` | asset-ready, smoke pending | DiffuEraser bridge | VideoDPO | VideoDPO winner | full-mask inpainting generated | full | full | false | offline | data | PSNR, SSIM, VBench, SBS, DPO diagnostics | Data/weight paths found; run one-sample generation smoke before full generation. |
| `official_videodpo_diffueraser_data_partialmask_loser_comp_k4` | asset-ready, smoke pending | DiffuEraser bridge | VideoDPO | VideoDPO winner | partial-mask inpainting + composite | partial K=4 | full | true | offline | data | PSNR, SSIM, VBench, SBS, DPO diagnostics | Cleanest partial-mask data-only ablation; wait for generator smoke. |
| `official_videodpo_diffueraser_data_partialmask_loser_nocomp_k4` | asset-ready, smoke pending | DiffuEraser bridge | VideoDPO | VideoDPO winner | partial-mask raw output | partial K=4 | full | false | offline | data diagnostic | PSNR, SSIM, VBench, SBS, DPO diagnostics | Reuses the same raw generation as comp; wait for generator smoke. |
| `official_videodpo_diffueraser_task_partialmask` | scaffold | DiffuEraser | generated partial-mask data | VideoDPO winner | partialmask comp loser | partial | partial | true | offline data | task | PSNR, SSIM, VBench, SBS, DPO diagnostics | First mask policy: same-mask. |
| `official_videodpo_diffueraser_youtubevos_partialmask_data` | path-confirmed scaffold | DiffuEraser / generator models | YouTube-VOS | YouTube-VOS clean/target clip | partial-mask generated loser | partial | partial | true first | offline | data source | PSNR, SSIM, VBench, SBS, DPO diagnostics | PAI train split confirmed under `ytbv_2019_full_resolution/train`; prompt policy still needs definition. |
| `official_videodpo_diffueraser_online_loser_generation` | future | TBD | TBD | TBD | generated during training | TBD | TBD | TBD | online | generation timing | TBD | Not first priority. |

## Offline / Online And Comp / No-Comp

First priority: `offline + comp`.

- Reproducible.
- Training speed is stable.
- Generation cost does not leak into DPO training time.
- Win and loser are identical outside the mask, giving the cleanest control variable.

Second priority: `offline + no-comp diagnostic`.

- Useful to determine whether compositing is necessary.
- Can introduce mask-outside color, texture, brightness, temporal, or background drift.
- Should not replace the comp main experiment.

Not first version: `online loser generation`.

- More diverse negatives, but costly and stochastic.
- Couples generation with training.
- Harder to debug than offline manifest-driven data.
## 2026-06-02 Target-Domain Boundary

VideoDPO experiments are now classified as bridge-domain engineering and
ablation work. The final target domains are YouTube-VOS and DAVIS.

Implications:

- Exp3 validates native VideoDPO DiffuEraser integration.
- Exp5 and Exp6 validate generated-loser data-only DPO and stabilization
  tricks; they are not final target-domain results.
- Exp7 validates partial-mask task plumbing; its VideoDPO partial-mask eval is
  diagnostic.
- Do not run VideoDPO partial-mask SFT warmup.
- Do not jump to Exp8 region loss before target-domain eval.
- D3 YouTube-VOS generated-loser data is background preparation for possible
  Exp9 only.

New rows:

| Experiment | Status | Domain | Data | Task | Eval | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `target_youtubevos_davis_existing_ckpts` | preflight script prepared | YouTube-VOS / DAVIS | existing target eval data | partial-mask / inpainting eval only | PSNR/SSIM/boundary/outside/temporal + qualitative | Mainline gate before Exp8/Exp9. |
| `d3_youtubevos_partialmask_loser_k4` | primary-comp gate ready / full readiness false | YouTube-VOS | generated loser D3 | data asset only | post-sync audit/readiness | 3,327 selected primary rows; secondary manifests are absent in slim sync, so full readiness is false but the first Exp9 primary-comp Stage1 gate is not blocked. |
| `exp9_target_domain_dpo_stage1_gate` | planned only | YouTube-VOS-derived D3 | D3 selected-primary comp repaired | partial-mask Stage1 DPO | target-domain eval gate | Do not launch until target eval shows bridge DPO does not transfer. |

## 2026-06-03 Target-Domain / Metric.py Policy Matrix

Metric boundary:

- VBench is only for video-generation or full-mask prompt-generation tasks.
- YouTube-VOS / DAVIS partial-mask video inpainting uses the project metric
  module. In this checkout no file named exactly `metric.py` exists; the
  existing metric backend is `inference/metrics.py`.
- New target-domain scripts must call `tools/run_inpainting_metric_eval.py`,
  which imports existing metric functions. Do not add new PSNR/SSIM formulas.

| Experiment | Status | Domain | Training data | Stage policy | Eval backend | Launch rule |
| --- | --- | --- | --- | --- | --- | --- |
| `target_youtubevos_davis_existing_ckpts_metricpy` | script prepared | YouTube-VOS / DAVIS | none | eval only | `inference/metrics.py` via wrapper | Run before Exp9; no VBench. |
| `exp9_youtubevos_d3_partialmask_wingap_lose025_stage1_gate1500` | launcher prepared; do not auto-run unless gate conditions hold | YouTube-VOS D3 | `selected_primary_comp.repaired.pai_paths.jsonl` or repaired manifest without H20 paths | Stage1 DPO only, no DPO Stage2 | target-domain metric wrapper | Start only after D3 primary readiness and target eval justify target-domain DPO. |
| `exp9_youtubevos_d3_partialmask_wingap_nolose_stage1_gate1000` | prepared only | YouTube-VOS D3 | same as Exp9 first gate | Stage1 DPO only, `lose_gap_weight=0.0` | target-domain metric wrapper | Fallback if lose-gap gate still shows loser degradation or artifacts. |

## 2026-06-03 H20 Complementary Matrix

| Experiment | Status | Host | Domain | Training data | Stage policy | Eval backend | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `new_exp6_prompt_length_audit` | prepared / running on H20 | H20 | VideoDPO bridge qual30 | existing Exp6 side-by-side | audit only | contact sheets + human labels | Tests long-prompt improvement hypothesis; not a final target-domain result. |
| `exp9_youtubevos_d3_nocomp_partialmask_wingap_lose025_stage1_gate1500_h20` | prepared for H20 launch | H20 GPUs 0-5 | YouTube-VOS D3 | selected-primary-nocomp H20 manifest | Stage1 DPO only, no DPO Stage2 | target-domain metric wrapper | Complements PAI D3-comp gate for comp-vs-nocomp decision. |

## 2026-06-03 Exp9 Comp / Nocomp Gate Monitor

| Experiment | Host | Status | Detected steps | Checkpoints | Eval readiness | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `exp9_youtubevos_d3_partialmask_wingap_lose025_stage1_gate1500` | PAI | stopped; invalid as Exp9 gate | about `4856/10000` at stop report time | stale output dir has `checkpoint-2000` and `checkpoint-4000` under `20260603_065327_exp5_d2_comp_k4_stage2_full` | not comparable | Wrapper printed Exp9 header, but stale env set `RUN_NAME=exp5_d2_comp_k4_stage2_full`, `MAX_STEPS=10000`, `CKPT_STEPS=2000`; do not use as gate1500. |
| `exp9_youtubevos_d3_nocomp_partialmask_wingap_lose025_stage1_gate1500_h20` | H20 | finished normally | `1500/1500` at 2026-06-04 01:08 CST | `checkpoint-500`, `checkpoint-1000`, `checkpoint-1500`, `last_weights` | ready | H20 report: `/home/nvme01/H20_Video_inpainting_DPO/reports/h20_exp9_nocomp_gate_monitor_report.md`; no DPO Stage2 and no VBench. |

## 2026-06-04 CST Exp9 Clean Gate / Eval Matrix

| Experiment | Host | Status | Eval status | Notes |
| --- | --- | --- | --- | --- |
| `exp9_youtubevos_d3_partialmask_wingap_lose025_stage1_gate1500` | PAI | clean gate running | pending completion | Clean launch confirmed D3 comp PAI manifest, Stage1-only, `max_steps=1500`, `ckpt_steps=500`, `ckpt_limit=5`; wait for `checkpoint-500/1000/1500` and `last_weights`. |
| `exp9_youtubevos_d3_nocomp_partialmask_wingap_lose025_stage1_gate1500_h20` | H20 | completed | D3/YouTube-VOS eval completed; DAVIS blocked | H20 D3 nocomp eval output: `logs/target_eval/exp9_d3_nocomp_gate_h20_20260604_023243`; `last_weights` improves mask PSNR/SSIM over base but has worse temporal/outside stability. Early checkpoints are accelerator states and need export before direct inference comparison. |

Exp9 eval rule:

- Evaluate only after both comp and nocomp gates expose comparable Stage1
  checkpoints.
- Use YouTube-VOS / DAVIS video inpainting metrics through
  `tools/run_inpainting_metric_eval.py` and `inference/metrics.py`.
- Do not use VBench for this inpainting comparison.

## 2026-06-08 Core Exp9 / Exp10 / Exp11 Matrix

The following rows supersede older A/B/C naming for the next PAI core sequence.
Historical Exp9 gates remain historical records and should not be confused with
this numeric sequence.

| Experiment | Status | Domain | Data | Stage policy | Loss focus | Eval |
| --- | --- | --- | --- | --- | --- | --- |
| `Exp9` / `exp9_logratio_gap_dpo_s1s2_2000_davis_pai` | prepared; default PAI launch | YouTube-VOS D3 + DAVIS val | D3 selected-primary comp repaired PAI manifest | Stage1 2000 -> DAVIS, Stage2 2000 -> DAVIS | log-ratio normalized gaps, clipped loser gap, still DPO | DAVIS raw6 no-PCM ProPainter-prior four-column videos + metrics |
| `Exp10` / `exp10_region_local_dpo_s1s2_2000_davis_pai` | H20 Stage1 running as explicit override; PAI prepared only | YouTube-VOS D3 + DAVIS val | same as Exp9 | Stage1 2000 -> DAVIS, Stage2 2000 -> DAVIS | Exp9 + normalized region-local MSE | same |
| `Exp11` / `exp11_flow_prior_consistency_dpo_s1s2_2000_davis_pai` | prepared but blocked even if GPUs are free | YouTube-VOS D3 + DAVIS val | same as Exp9 | do not train until audit passes | Exp10 + flow/prior/boundary consistency | same after audit |

No-lose-gap remains diagnostic only. Removing lose-gap makes the loser branch
weak and damages the pairwise DPO interpretation, so it is not a main method.

2026-06-08 runtime note:

- H20 Exp10 is running on GPU4-7 with the H20-safe fp32/no-split profile because
  bf16/split paths can hit SIGFPE on H20. Latest observed Stage1 step: 210.
- PAI GPU0-3 appeared lightly occupied while Exp9 used GPU4-7, but that
  availability does not permit Exp11 training. Exp11 remains blocked by the
  train-time flow/prior implementation audit.

## Exp20 Status Addendum (2026-06-20)

Exp20 fast search + equal-step budget completed. Best current locked-dev/equal-step candidate is EQ_BF07 (fixed_image_px radius 28, boundary weight 5.0, legacy_global_weighted_mean), PSNR 29.393079. It remains below TARGET_DEV_PSNR 29.523336 and has mixed perceptual/temporal tradeoffs, so status is COMPLETED_NEGATIVE for this budget and no long training / Stage2 / DAVIS50 / YouTubeVOS100 final eval was launched.
## 2026-06-24 Exp26 Update

Exp26 Probe4 remains passed. Existing Gate16 outputs were reclassified without
replacement: `medium-hard=15`, `trivial-bad=1`, `technical-invalid=0`.
Numeric gate criteria pass, but interactive video playback remains pending, so
Gate64 and DPO training were not started.

## 2026-06-24 Exp26 Gate64 Protocol

| Experiment | Status | Notes |
| --- | --- | --- |
| Exp26 VideoPainter v2 Gate64 | `GATE64_PROTOCOL_LOCKED_PENDING_PAI_GENERATION` | Mixed-mask protocol `vp2_mixed_br_mask_v1` locked; 64 scene-disjoint VOR-BG sources; no Gate64 generation or DPO training launched yet. |

## 2026-06-24 Exp26 Gate64 Generation Launcher

| Experiment | Status | Notes |
| --- | --- | --- |
| Exp26 VideoPainter v2 Gate64 generation | `GATE64_GENERATION_IMPLEMENTED_PENDING_PAI_RUN` | Fixed pre-run mixed-mask implementation bug; added selective VOR-BG extraction and official 49F generation launcher. No DPO training launched. |
## 2026-06-25 Pre-Maintenance Persistence Gate

- Exp25 status: `BLOCKED_NAS_PERMISSION`
- Exp26 status: `BLOCKED_NAS_PERMISSION`
- Reason: PAI SSH user `hj` cannot write to
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch`; required
  pre-maintenance persistence markers were not created.
- Effect: no new Exp25 root-cause matrix, Exp26 Gate64 review/source repair, or
  Exp27 true-model GPU task was launched after the blocker.

## 2026-06-25 Exp26 Gate64 Visual Review

| Experiment | Status | Notes |
| --- | --- | --- |
| Exp26 VideoPainter v2 Gate64 review | `GATE64_EVIDENCE_REVIEW_COMPLETE_MP4_PLAYBACK_PENDING_POOL_NOT_DATA_READY` | Evidence-reviewed 56 generated outputs: 31 medium-hard, 16 hard-plausible, 1 too-close, 8 trivial-bad. 47 eligible rows require balanced manifest construction before any DPO micro-training. |

## 2026-06-25 Exp26 Gate64 Static-Frame Repair

| Experiment | Status | Notes |
| --- | --- | --- |
| Exp26 VideoPainter v2 Gate64 source repair | `GATE64_PRIMARY32_DRAFT_MP4_PLAYBACK_PENDING_DPO_BLOCKED_BY_NAS_EXPERIMENT_PERMISSION` | The 8 rows previously marked formal failures were re-audited with timestamp/frame-index evidence and classified as static-pixel duplicates, not 49F read failures. All 8 were regenerated and evidence-reviewed: 6 medium-hard, 2 hard-plausible. Final Gate64: 64/64 formal-valid, 55 eligible, 9 rejected. A balanced 32-row comp-loser primary draft manifest is locked at `exp26_videopainter_dpo_v2/manifests/vp2_gate64_primary32_visual_reviewed_comp.jsonl`, but strict mp4 playback remains pending before `DATA_READY`. DPO micro-training remains blocked until `hj` can write `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp26_videopainter_dpo_v2`. |

## 2026-06-25 PAI Post-Maintenance Permission Recovery

| Experiment | Status | Notes |
| --- | --- | --- |
| Exp26 VideoPainter v2 | `PAI_POSTMAINTENANCE_PERMISSIONS_RECOVERED` | Exp26 NAS experiment/autoresearch output roots are now writable to `hj`; next allowed milestone is strict Gate64 temporal review, not regeneration or long training. |

## 2026-06-25 Exp26 Gate64 Final Temporal Review

| Experiment | Status | Notes |
| --- | --- | --- |
| Exp26 VideoPainter v2 Gate64 primary data | `GATE64_DATA_READY` | All 64 Gate64 outputs received final temporal evidence review. Final pool: 37 medium-hard, 18 hard-plausible, 1 too-close, 8 trivial-bad, 0 technical-invalid. The final balanced primary-32 comp-loser manifest is `exp26_videopainter_dpo_v2/manifests/vp2_gate64_primary32_final.jsonl` with SHA256 `82f661f2f30a581a213972533817624217eabb97eba7aaeedc00ee2109e4e716`; search/shadow scene overlap is 0; all primary videos decode as 49 frames. |

| Exp26 VideoPainter primary-32 L0/L1 | `TECHNICAL_PASS` | Real 49F final-primary32 DPO batch and one optimizer step passed on PAI GPU0: DPO loss finite, policy grad norm 14.3799, reference grad false, policy delta 1.6733, reference delta 0, strict reload max diff 0. No 10-step/50-step quality claim yet. |

| Exp26 VideoPainter primary-32 10-step | `TRAINING_PASS` | 10-step DPO micro gate passed on search-dev with dense temporal evidence review. Step10 comp vs step0: PSNR +0.977252, SSIM +0.032641, LPIPS -0.004499, Ewarp -1.301457, mask PSNR +0.975192, boundary PSNR +5.082206; no global collapse or systematic new visual artifact found in 32/32 dense evidence/crop sheets. Conditional 50-step gate is allowed; RC-FPO and long training remain not started. |

| Exp26 VideoPainter primary-32 50-step | `VIDEOPAINTER_ADAPTER_POSITIVE` / `TRAINING_PASS` | 50-step DPO micro gate passed on locked search-dev: step50 comp vs step0 PSNR +4.816168, SSIM +0.087883, LPIPS -0.044059, Ewarp -7.055122, strict mask PSNR +4.942246, boundary PSNR +12.111889; PSNR bootstrap probability(delta>0)=1.0. Manual temporal/crop review covered 32/32 rows with no gate-blocking systematic artifact. This remains search-dev micro evidence only, not `SCIENTIFIC_POSITIVE`; RC-FPO and 100-step+ training are still not started. |
## 2026-06-26 Exp26 Shadow-Dev Confirmation

| Experiment | Status | Evidence | Next Restriction |
| --- | --- | --- | --- |
| Exp26 VideoPainter DPO v2 | `VIDEOPAINTER_SHADOWDEV_CONFIRMED` | Fixed Step50 vs fixed Step0 on independent 32-row shadow-dev: strict mask PSNR `+5.186942`, boundary PSNR `+12.175098`, LPIPS `-0.040142`, Ewarp `-8.378847`, TC `+0.004378`, VFID/FVD-style `-0.031428`; 32/32 visual review; seed robustness pass `3/3`. | No 100-step+, no RC-FPO, no universal/final SOTA claim before external cross-dataset benchmark. |
| Exp26 VideoPainter post-confirmation sanity | `EXP26_POSTCONFIRMATION_SANITY_AUDIT_PASSED` | Re-read checkpoint identity, search/shadow reports, leakage audit, visual review, seed robustness, and dynamics. Evidence remains internally consistent; no unexpected GT leakage or comp mask overwrite was found. | Proceed only to external 49F validation / evidence pack / compatibility audit; still no training. |
| Exp26 external 49F inventory | `EXP26_EXTERNAL_49F_INVENTORY_COMPLETE` | Scanned `2024` local candidate directories; found `54` valid DAVIS-derived clean 49F sources and locked `32` rows at SHA256 `be118a7ce7d462bda6c339053d0c112994c8da7fab6cf00a4ee5dae87b628e5a`. | Preregister masks/seeds/protocol before any Step0/Step50 external inference. |
| Exp26 external validation preregistration | `EXP26_EXTERNAL_VALIDATION_PREREGISTERED` | Locked 32 exact-49F DAVIS-derived rows, deterministic mixed masks, first-frame GT, seed `20260619`, mask seed `20260623`, and fixed Step0-vs-Step50 protocol. Preregistered manifest SHA256 `69ecd96d4b25da702229df2d45bf1343ad5e7ef5385cbd32d24ce61644e4bc2c`; mask manifest SHA256 `f646792469f53a8122fe341be5988344ba7b32d33b3a53593d558e227aed138b`. | External inference can start from the locked manifest; Step10/30 remain trajectory-only and cannot replace Step50. |
| Exp26 external validation generation | `EXP26_EXTERNAL_GENERATION_COMPLETE` | Fixed Step0/Step10/Step30/Step50 outputs completed `32/32` rows each under the post-confirmation external-validation root. Leakage audit covered `128` checkpoint/sample rows and found `0` unexpected winner-copy cases. | External metrics and full video review remain required; no checkpoint reselection or retraining is allowed from this split. |
| Exp26 external validation metrics | `EXP26_EXTERNAL_VALIDATION_NOT_CONFIRMED` | Fixed Step50 vs fixed Step0 on external frame1-48 comp: strict mask PSNR delta `-2.610576`, win rate `0.218750`, bootstrap probability improved `0.006500`; whole comp PSNR `-2.563047`; LPIPS worsened `+0.002466`. TC/VFID-style was mixed: TC `0.961672` vs `0.962637`, VFID `0.397402` vs `0.420941`. | External video review should classify failure modes; Step10/30 cannot replace Step50, and no 100-step/retraining is authorized. |

| Exp26 external validation visual review | `EXP26_EXTERNAL_VIDEO_REVIEW_COMPLETE` / `EXP26_EXTERNAL_VALIDATION_NOT_CONFIRMED` | Reviewed `32/32` external DAVIS-derived Step0-vs-Step50 rows with blind/informed evidence pages and crop sheets. Step50 was slightly better in `3`, tied in `5`, and worse in `24`; `29` rows showed Step50-specific local artifacts. This strengthens the external negative result and forbids checkpoint reselection or further VideoPainter training from this split. |

| Exp26 VideoPainter evidence pack | `EXP26_RESULT_PACK_COMPLETE` | Built a 30-case discussion pack spanning search-dev examples, shadow-dev confirmed positives/ties/failures, and external DAVIS-derived limited positives/failures. The pack strengthens cross-backbone evidence for DiffuEraser + VideoPainter on VOR-BG but explicitly preserves `EXP26_EXTERNAL_VALIDATION_NOT_CONFIRMED` for external generalization. |

| Exp26 third-model compatibility audit | `EXP26_THIRD_MODEL_COMPATIBILITY_AUDIT_COMPLETE` | Static audit found no third model currently `TRUE_DPO_ADAPTER_READY`. CoCoCo is the lowest-risk future adapter candidate after weights/dependency/native-parity gates; MiniMax/ProPainter are better immediate baseline or loser-generator targets. No third-model training launched. |

| Exp29 MiniMax + EffectErase OR adapter feasibility | `EXP29_READBACK_AND_SCAFFOLD_CREATED` | New isolated branch/worktree created from Exp26 post-confirmation state. Scope is repo/weight/code feasibility, optional inference smoke only if verified assets exist, and optional trainable-forward gates only after smoke. No long training, RC-FPO, VideoPainter 100-step, or left CLI modification. |
| Exp29 MiniMax repo/weight audit | `MINIMAX_REPO_READY` / `MINIMAX_WEIGHTS_READY` | MiniMax repo and PAI/NAS weights are available. It is eligible for isolated inference smoke, but no smoke, trainable-forward gate, or adapter step has run yet. |
| Exp29 EffectErase repo/weight audit | `EFFECTERASE_REPO_READY` / `EFFECTERASE_BLOCKED_NO_WEIGHTS` | EffectErase repo and generic Wan training utilities are present, but official `EffectErase.ckpt` and Wan2.1-Fun InP assets were not found. It remains blocked before smoke and is VOR diagnostic/baseline only. |
| Exp29 MiniMax inference smoke | `MINIMAX_INFERENCE_SMOKE_PASSED_WITH_VISUAL_QUALITY_RISKS` | MiniMax generated 4/4 fixed smoke outputs. One row is a medium-hard OR candidate; three rows are trivial-bad due to residual object or artifacts. This is a technical smoke pass, not robust OR baseline proof. |
| Exp29 MiniMax trainable forward | `MINIMAX_TRAINABLE_FORWARD_PASSED` | Native flow forward used `epsilon - z0` velocity target, finite MSE loss `0.0171425510`, grad norm `0.7473063172`, 461 gradient tensors, and clean state_dict key identity. Next eligible step is zero-gap / one-step / 10-step micro gate. |
| Exp29 MiniMax adapter gates | `MINIMAX_10STEP_PARETO_MIXED` | Zero-gap and one-step strict reload passed. Conservative 10-step micro completed without NaN after fp16 AdamW was found unstable, but heldout Step10 was visually tied with Step0. MiniMax remains `ADAPTER_POSSIBLE_NEEDS_MORE_WORK`, not a confirmed third backbone. |
| Exp29 continuation | `EXP29_CONTINUATION_READBACK_COMPLETED` | Exp29 continuation reread branch/PRD/registry/reports and left CLI state before GPU work. MiniMax remains pending medium-hard data and recipe gates; EffectErase remains pending weight recovery. |
| Exp29 MiniMax 10-step failure analysis | `MINIMAX_10STEP_FAILURE_ANALYZED` | The 10-step heldout tie is explained by an intentionally tiny stable SGD update, mostly trivial-bad loser quality, and a 2-row heldout set. Do not extend the same recipe; require data-quality and recipe gates first. |
| Exp29 MiniMax medium-hard data gate | `MINIMAX_DATA_YIELD_INSUFFICIENT` | 96 reviewed MiniMax candidates produced 27 eligible candidates but only 9 eligible scene groups. The required scene-disjoint train16/heldout16 split cannot be built; optimizer recipe search and 30-step confirmatory micro are blocked. |
| Exp29 EffectErase weight recovery | `EFFECTERASE_WEIGHTS_READY` | Official EffectErase checkpoint and Wan2.1-Fun InP assets are now present under the Exp29 NAS cache, 20G total, 53 files, 19/19 SHA entries OK. Next allowed step is inference smoke only; no adapter or scientific-positive claim is unlocked by weight recovery alone. |
| Exp29 EffectErase smoke preregistration | `EFFECTERASE_SMOKE_PREREGISTERED` | Locked 6 diagnostic VOR rows, balanced REAL/BLENDER and small/medium/large masks, at SHA256 `54fd62a97fa69f2f17590488136d426cee77de0ed02548c46a83d8818be2b137`. Protocol is 17-frame 832x480 removal smoke with raw output primary. All rows are diagnostic-only and not eligible for training; no inference or adapter claim yet. |
| Exp29 continuation v3 readback | `EXP29_CONTINUATION_V3_READBACK_COMPLETED` | Confirmed Exp29 HEAD `972deab321a518638102a1ace6ed87a13456a261`, EffectErase weights-ready/preregistered state, MiniMax data-yield-insufficient blocker, and left CLI protection. No GPU inference, MiniMax recipe, 30-step micro, or left-side action launched by readback. |
| Exp29 EffectErase smoke input materialization | `EFFECTERASE_SMOKE_INPUTS_BLOCKED` | Materialized 5/6 locked diagnostic rows as 17-frame 832x480 mp4s. The locked row `REAL_ENV249_00103_004_04` has an empty mask, so the preregistered six-row smoke is blocked before inference. No row replacement or inference launch occurred. |
| Exp29 EffectErase command dry-run | `EFFECTERASE_COMMAND_READY` | Dedicated Exp29 venv imports official EffectErase inference with compatible `transformers==4.51.3`; official script supports `--num_frames 17`, CFG, steps, and seed arguments. Full inference is still blocked by the preregistered empty-mask input row. |
| Exp29 MiniMax expanded source-pool plan | `MINIMAX_EXPANDED_GENERATION_BLOCKED` | Current Exp25 semantic audit has only 64 rows and 31 valid unused rows after excluding the prior 32-source gate, far below the 96/128-source first-pass requirement. No generation, recipe search, 30-step micro, or training launched. |
| Exp29 continuation v4 readback | `EXP29_CONTINUATION_V4_READBACK_COMPLETED` | Confirmed Exp29 HEAD `5e20149363b16f4728016260ff3e6d79dace299d`, old EffectErase empty-mask blocker, MiniMax full-VOR source-audit requirement, and left CLI GPU1-GPU4 reservation. No EffectErase inference, MiniMax generation, recipe, 30-step, training, or RC-FPO launched by readback. |
| Exp29 EffectErase smoke v2 preregistration | `EFFECTERASE_SMOKE_V2_PREREGISTERED` | Preserved old manifest, rejected empty-mask `REAL_ENV249_00103_004_04`, replaced it with valid non-empty-mask `REAL_ENV248_00118_005_03`, and locked v2 SHA256 `b16a0007a22f190bb7894a673092063efb5dd2eda26dbd53737cdc987d9d4f36`. No inference or adapter claim yet. |
| Exp29 EffectErase smoke v2 materialization | `EFFECTERASE_SMOKE_V2_INPUTS_READY` | Materialized 6/6 v2 diagnostic rows as 17-frame 832x480 condition/winner/mask mp4s under the v2 output root. All masks are non-empty in 17/17 frames; VOR-Eval and training eligibility remain false. No inference launched yet. |
| Exp29 EffectErase official smoke v2 | `EFFECTERASE_SMOKE_BLOCKED_FRAME_COUNT_INCOMPATIBLE` | Official script loads model assets after PYTHONPATH fix but does not pass `args.num_frames` into `WanRemovePipeline`; the pipeline uses 81-frame noise latent time 21 while v2 locked inputs encode to time 5. No output video or baseline-ready claim. |
| Exp29 MiniMax full-VOR source audit | `MINIMAX_FULL_VOR_SOURCE_AUDIT_READY` | Read Exp25 full VOR Train metadata index read-only. After excluding previous MiniMax source32 and EffectErase smoke rows, 1,417 candidate scene groups remain. Locked 192 scene-disjoint candidates, REAL/BLENDER = 96/96, manifest SHA256 `16e128282da110eeefd6cb56a517c8b6de82e42a5241c9b845e01315d9800f10`. Mask/effect/motion labels remain pending materialization; no generation, recipe, 30-step, or training launched. |
| Exp29 MiniMax expanded data-yield v2 | `MINIMAX_EXPANDED_DATA_YIELD_INSUFFICIENT` | Seed A reviewed 96 candidates and conditional seed B reviewed 32 near-miss candidates. Combined attempts produced 24 medium-hard, 2 hard-plausible, 14 too-close, 77 trivial-bad, and 11 technical-invalid rows, but only 26 eligible unique scene groups after merge. Train16+heldout16 cannot be built, so recipe/30-step/training remain stopped. |
| Exp29 continuation v5 readback | `EXP29_CONTINUATION_V5_READBACK_COMPLETED` | V5 scope locked: EffectErase must switch to official 81-frame smoke, MiniMax may only do top-up data-yield before any 10-step recipe, and left CLI remains read-only protected. No GPU inference or training launched by readback. |
| Exp29 EffectErase official 81F source audit | `EFFECTERASE_OFFICIAL81_PREREGISTERED` | Locked 8 diagnostic-only official-81F EffectErase smoke rows from existing Exp25 full-VOR metadata/exact extraction caches, SHA256 `706cb09286fd8528d7efbbb91eb89673a9ec7ce61b0047e6b3b2e8ea4c9b1fb3`. Codex opened 8/8 preview sheets and found no source/mask/frame-order invalidity. No inference, baseline-ready claim, or adapter claim yet. |
| Exp29 EffectErase official 81F input materialization | `EFFECTERASE_OFFICIAL81_INPUTS_READY` | Materialized 8/8 locked rows as 832x480 condition/winner/mask MP4s with 81 decoded frames each. Codex opened 8/8 materialized preview sheets and found no resize/encoding input invalidity. No inference, baseline-ready claim, or adapter claim yet. |
| Exp29 EffectErase official 81F command validation | `EFFECTERASE_OFFICIAL81_COMMAND_READY` | Dry-run validated official EffectErase command/import/assets/inputs for 8/8 official-81F rows. Commands use `--num_frames 81`; no full inference, baseline-ready claim, or adapter claim yet. |
| Exp29 EffectErase official 81F inference smoke | `EFFECTERASE_OR_BASELINE_READY` | Official 81F inference completed 8/8 diagnostic VOR rows on right-side GPU0. All raw outputs decode as 81 frames at 832x480. Codex opened 8 temporal pages and 8 crop pages: target/effect removal succeeded 8/8 with no black/purple collapse. Project metric means: whole PSNR 27.416948, LPIPS 0.085822, mask PSNR 25.778614, boundary PSNR 25.696018, Ewarp 1.766501. This is OR strong baseline / diagnostic readiness only; no true adapter, DPO, or universal-adapter claim. |
| Exp29 EffectErase trainable-forward audit | `EFFECTERASE_BASELINE_ONLY_FOR_NOW` | Official removal pipeline audit found no removal-specific trainable forward/training loss exposed by `WanRemovePipeline`; the generic Wan training path is not equivalent to the EffectErase removal adapter path. Zero-gap, one-step, 10-step, DPO, and RC-FPO were not run. EffectErase remains an OR strong baseline/diagnostic, not true adapter evidence. |
| Exp30 VOR-OR multi-model MiniMax readback | `EXP30_READBACK_COMPLETED` | New isolated Exp30 branch/worktree created from Exp29 HEAD `6bc6c67`. Scope is VOR-OR multi-model medium-hard pool construction, MiniMax quality-positive micro gate, DiffuEraser VOR-OR Stage1/Stage2 micro validation, and paper-ready three-backbone evidence planning. PAI GPU readback found no compute processes, but left CLI locks reserve GPU1/GPU2/GPU3/GPU4. No GPU task, inference, training, RC-FPO, or left-side action launched. |
| Exp30 paper three-backbone positioning | `EXP30_THREE_BACKBONE_POSITIONING_LOCKED` | Roles locked before pool generation: DiffuEraser is original backbone and VOR-OR micro target; VideoPainter is second VOR-BG BR/inpainting backbone; MiniMax is flow-style third-backbone candidate pending quality-positive heldout micro; EffectErase is OR strong baseline/diagnostic only. Universal-adapter and all-models-supported language remain forbidden. |
| Exp30 VOR-OR source pool audit | `VOR_OR_SOURCE_POOL_BLOCKED` | Existing exact extraction caches yielded only 80 usable scene groups after exclusions, with reserve 0 and REAL/BLENDER = 71/9. Codex opened 10/10 preview pages covering all 80 rows; source triplets are visually aligned and mask/affected valid, but the pool fails the requested 128 source + 128 reserve target and balance requirements. No smoke, Gate128, MiniMax adapter, DiffuEraser VOR-OR micro, GPU task, or training launched. |
| Exp30 continuation v2 full VOR index readback | `EXP30_CONTINUATION_V2_READBACK_COMPLETED` | Full metadata index with 57,751 rows located on PAI; previous 80-row gate failure was an exact-extraction-cache subset issue, not a full VOR shortage. No GPU task or left-side action launched. |
| Exp30 full VOR valid triplet index recovery | `FULL_VOR_VALID_TRIPLET_INDEX_READY` | Exp25 full metadata index verified with 57,751 rows, 57,750 valid triplets after quarantining `BLENDER_RIVER007_00001`, 1,449 scene groups, and exact FG_BG/BG/MASK basename pairing. No archive scan or extraction. |
| Exp30 source-pool v2 sampling | `VOR_OR_SOURCE_POOL_V2_READY` | Built metadata-only primary128/reserve128/reserve2 pools from the full VOR index. Primary is BLENDER/REAL 64/64; reserve is 20/108; reserve2 is 0/128. Mask/effect labels remain unknown. This unlocks smoke16 only, not Gate64/training/data-ready claims. |
| Exp30 smoke16 v2 preregistration | `EXP30_SMOKE16_V2_PREREGISTERED` | Locked 16 source groups from source-pool v2 with BLENDER/REAL 8/8 balance and manifest SHA256 `1871f8e1aa23579425a87661040f91a992e934492aaa98c196f924ff21990ca3`. No model outputs, metrics, video review, Gate64, or training have run yet. |
| Exp30 smoke16 v2 pre-inference repair | `EXP30_SMOKE16_V2_MANIFEST_REPAIRED_PRE_INFERENCE` | First materialization found one short decoded row and two empty-mask rows. Repaired before any model output review with deterministic same-source-type, scene-disjoint source-pool-v2 replacements. Final smoke16 manifest has 16 scene-disjoint rows, BLENDER/REAL 8/8, SHA256 `7e8cfd1b672b17b131476c9dd82804841d22d7450adf26301cf9ae8ff83f7f76`. Still no smoke pass, Gate64, training, or data-ready claim. |
| Exp30 smoke16 v2 final materialization | `EXP30_SMOKE16_V2_MATERIALIZED` | Final repaired smoke16 rows materialized successfully: 16/16 rows, 17 frames, 512 x 512, failed rows 0, BLENDER/REAL 8/8, manifest SHA256 `72be9884335fef61926c307c66878fdc05dec85e9be4da28ab1547db98f8c26d`. This unlocks smoke16 candidate generation only, not smoke pass, Gate64, training, or scientific claims. |
| Exp30 multi-model OR smoke16 v2 | `MULTIMODEL_OR_SMOKE16_V2_BLOCKED` | Generated/reviewed 32 non-EffectErase candidates: 16 controlled-corruption and 16 MiniMax official. Technical-valid = 32/32; total usable = 9/32, but controlled-corruption usable fallback = 5/16, below the preregistered >=6/16 requirement. Gate64, MiniMax adapter recipe/training, and DiffuEraser VOR-OR micro remain stopped. |
| Exp30 continuation v3 readback | `EXP30_CONTINUATION_V3_READBACK_COMPLETED` | Confirmed Exp30 HEAD `bd8777274dfe898dc9278cadcc1dd971536a5e2c` and reread PRD/registry/reports/code. Smoke16 v2 remains blocked by controlled-corruption usable fallback 5/16 < 6/16 despite 32/32 technical-valid non-EffectErase candidates. Exp31 is running on GPU1, Exp33 on GPU3, and left CLI locks reserve GPU1-GPU4; Exp30 must use only fresh eligible non-reserved GPUs. No new generation, Smoke32, Gate64, adapter gate, or training launched. |
| Exp30 smoke16 v2 failure analysis | `SMOKE16_V2_FAILURE_ANALYZED` | Analyzed all 32 v2 candidates without new generation. Controlled corruption is mostly blocked by temporal discontinuity from an over-aggressive single profile: 11/16 trivial-bad and 5/16 usable. MiniMax is mostly blocked by outside damage and temporal flicker/instability: 12/16 trivial-bad and 4/16 usable. V3 fixes must be preregistered: softer temporal controlled profiles and verified DiffuEraser/ProPainter candidate families. |
| Exp30 controlled corruption v3 plan | `CONTROLLED_CORRUPTION_V3_PLAN_LOCKED` | Locked bounded profiles CC-v3-A/B/C plus reserve D. Smoke16 v3 controlled schedule is capped at 24 candidates: B for all 16 sources, A for six temporal-discontinuity repair sources, and C for two affected-soft sources. Success target is >=8/16 usable controlled source coverage with no systematic outside damage. No generation or training launched. |
| Exp30 DiffuEraser/ProPainter candidate audit | `NEW_GENERATORS_SMOKE2_PENDING` | DiffuEraser has verified Exp25 `DE-B` quality evidence, but Exp30 current wrapper identity is not the verified no-PCM overlay path. ProPainter assets are ready on PAI. Next gate is wrapper port + two-sample generator smoke; Smoke16 v3, Smoke32, Gate64, and training remain stopped. |
| Exp30 verified generator wrapper port | `EXP30_VERIFIED_GENERATOR_WRAPPERS_PORTED_SMOKE2_PENDING` | Added Exp30-isolated DiffuEraser no-PCM overlay wrapper and verified-generator smoke runner. No GPU smoke or model output yet; Smoke16 v3/Gate64/training remain blocked until smoke2 passes. |
| Exp30 verified generator smoke2 | `NEW_GENERATORS_SMOKE2_PARTIAL_PASS` | ProPainter and DiffuEraser no-PCM generated 2/2 rows each. Codex reviewed 4/4 sheets: 2 too-close, 1 hard-but-plausible, 1 medium-hard, 0 final trivial-bad. This supports adding these generators to Smoke16 v3 only; Smoke32/Gate64/training remain stopped. |
| Exp30 controlled corruption v3 generator | `CONTROLLED_CORRUPTION_V3_GENERATOR_IMPLEMENTED` | Added the preregistered CC-v3-A/B/C controlled-corruption generator and deterministic primary controlled selection. No Smoke16 v3 generation, Smoke32, Gate64, adapter gate, or training launched by this implementation milestone. |
| Exp30 controlled corruption smoke16 v3 | `CONTROLLED_CORRUPTION_V3_READY` | Generated 24/24 v3 controlled candidates and reviewed all candidate/primary pages. Primary controlled view: 16/16 technical-valid, 13 medium-hard, 3 trivial-bad, outside-fail 0. This repairs only the controlled fallback subgate; aggregate Smoke16 v3/Gate64/training remain stopped. |
| Exp30 multi-model OR smoke16 v3 | `MULTIMODEL_OR_SMOKE16_V3_PASS` | 64/64 candidates technical-valid and best-per-source usable 13/16. The pass is driven by controlled corruption v3; ProPainter contributes 2 usable rows and DiffuEraser 0 usable rows on this fixed set. Only Smoke32 is unlocked next; Gate64/training remain stopped. |
| Exp30 smoke32 v3 preregistration | `EXP30_SMOKE32_V3_PREREGISTERED` | Locked 16 new confirmation source groups, disjoint from Smoke16 and BLENDER/REAL 8/8. No extraction, model output, Gate64, adapter gate, or training launched. |
| Exp30 smoke32 v3 materialization | `EXP30_SMOKE32_V3_MATERIALIZED` | Selective extraction wrote 48/48 target VOR members with 0 missing/unsafe entries. Materialized 16/16 Smoke32 rows at 17 frames and 512x512, failed rows 0, manifest SHA256 `320bb89ba16fb61a005e533ab319a2f4fb9ee6362cb8c269d4f2f0223a3e2ce9`. Only Smoke32 candidate generation is unlocked; Gate64/training remain stopped. |

| Exp30 Smoke32 V3 multi-model OR confirmation | `MULTIMODEL_OR_SMOKE32_V3_PASS` | 64/64 non-EffectErase candidates technical-valid; 14 usable candidates; best-per-source usable 10/16; controlled v3 usable coverage 8/16; usable families are controlled v3, MiniMax official v3, and ProPainter. DiffuEraser no-PCM remains 0 usable in this confirmation split. This unlocks limited Gate64 pool preparation only; no data-ready or adapter-training claim yet. |
| Exp30 Gate64 V3 preregistration | `EXP30_GATE64_V3_PREREGISTERED` | Locked 64 source groups after Smoke16/Smoke32 v3 passed; BLENDER/REAL 32/32; SHA256 `c4a0f5e07ef75aae57c9b40010f7fec85d10d5aa6c26a8056e1079d807bcf7f2`. No model output, visual selection, or training yet. |
| Exp30 Gate64 V3 pre-inference repair | `EXP30_GATE64_V3_MANIFEST_REPAIRED_PRE_INFERENCE` | Initial materialization found 9 empty-mask rows before model output; replacements keep final 64 scene groups and BLENDER/REAL 32/32. Final manifest SHA256 `c2da063118934f0b03d13d88015cfc1cc57e881aca257307ca42de20cc944eb0`. |
| Exp30 Gate64 V3 final materialization | `EXP30_GATE64_V3_MATERIALIZED` | Final repaired Gate64 source set materialized 64/64 rows at 17 frames and 512x512; failed rows 0; BLENDER/REAL 32/32; SHA256 `a32d42b9d5f9894e3e4c8f177b04e8d98271670b864f2388f72a5cb98dc02d13`. Candidate generation only is unlocked. |
| Exp30 Gate64 V3 multi-model OR pool | `VOR_OR_GATE64_MULTIMODEL_POOL_READY` | Aggregated 256 Gate64 candidates and selected 50 usable primary pairs: 48 medium-hard and 2 hard-plausible. Train32 SHA256 `1eda205d2dc48714269f30eb390d959549387a778e6438267e6aba087ba14196`; heldout16 SHA256 `84c231ded930d740bf299b27c2a6b1e95d7decdb3665051371c5df90ae9f2ade`; train/heldout scene overlap 0. This unlocks only the preregistered MiniMax 10-step adapter gate; no training or scientific-positive claim yet. |
| Exp30 MiniMax Gate64 adapter 10-step | `MINIMAX_ADAPTER_RECIPE_NOT_READY` | Frozen and EMA MiniMax flow-target recipes both passed zero-gap and one-step strict reload on Gate64 train32/heldout16, but Step10 did not produce a heldout quality gain. Codex opened 8 combined review pages; visual better 0/32, tie 32/32. Mean heldout mask/boundary/outside PSNR deltas were negative for both recipes. Stop before 30-step or long training. |
| Exp35 MiniMax flow-DPO rescue readback | `EXP35_READBACK_COMPLETED` | New isolated Exp35 branch/worktree created from Exp30 HEAD `f69688f`. Readback classifies the MiniMax no-change issue as recipe/update-state/noise-regime suspected rather than data or basic plumbing: full transformer was trainable, LR was `5e-7`, utility stayed near `0.5`, Step10 delta probe was about `2.7e-7`, and no bad-noise miner was used. No GPU inference, training, 30-step, RC-FPO, or protected-lane action launched. |
| Exp35 MiniMax no-change forensic audit | `MINIMAX_NOCHANGE_CAUSE_UTILITY_SCALE_TOO_WEAK` | Audited Exp30 frozen/EMA checkpoint-0 vs checkpoint-10 and heldout frames without training. Keys matched 461/461, missing/unexpected 0/0, and Step10 outputs were not byte-identical to Step0, so this is not an obvious fallback. Output movement was sub-perceptual and utility stayed essentially constant near 0.5; next gate is inference-sensitivity positive-control. |
| Exp35 MiniMax inference sensitivity | `MINIMAX_INFERENCE_SENSITIVITY_PASS` | No-training positive-control on 4 rows. Step0 identity replay was deterministic, while a temporary Exp35-only perturbation of 16 MiniMax transformer tensors produced measurable but visually subtle output movement. Inference path is sensitive to weights; continue to scope/objective diagnostics before any recipe training. |
| Exp35 MiniMax trainable-scope audit | `MINIMAX_TRAINABLE_SCOPE_CURRENT_OK` | No-training safetensors audit found 461 MiniMax transformer tensors, 1.127B represented parameters, and 0 LoRA/adapter tensors. Exp30 used full-transformer scope and Milestone B proves inference consumes those weights; the no-change failure is not a too-small/ignored adapter scope. |
| Exp35 MiniMax winner-SFT positive-control | `MINIMAX_POSITIVE_CONTROL_PASS_HELDOUT_QUALITY_NEGATIVE` | Bounded 10-step supervised winner reconstruction on PAI GPU6 reduced train loss and moved heldout outputs, proving trainability/checkpoint sensitivity. Heldout mask/boundary PSNR deltas were negative for all recipes, and Codex reviewed 12/12 temporal strips with no quality-positive rows; LR 1e-4 produced new artifacts in 4/4 rows. No 30-step or third-backbone positive claim is unlocked. |
| Exp35 MiniMax bad-noise / hard-timestep miner | `MINIMAX_BAD_NOISE_STATES_READY` | Frozen Step0 miner scanned train32/heldout16 with `K_noise=4` and timesteps `0.15/0.35/0.55/0.75`, producing 768 candidate-state records and fixed hard_state_A/B/C manifests. This is state preregistration for bounded 10-step recipe testing only; it is not a model-quality result and does not unlock 30-step. |
| Exp35 MiniMax rescue recipe preregistration | `MINIMAX_RESCUE_RECIPES_PREREGISTERED` | Locked three active 10-step recipes: R1 frozen hard-noise Linear-DPO, R2 EMA hard-noise Linear-DPO, and R3 winner-anchor hybrid. All use `hard_state_A`, LR `1e-5`, utility scale `10`, and fixed Gate64 train32/heldout16. R4 SDPO-safe hybrid is inactive. No training or 30-step launched. |
| Exp35 MiniMax rescue 10-step recipe gate | `MINIMAX_RESCUE_RECIPE_NOT_READY` | R1/R2/R3 10-step recipes completed on locked train32/heldout16 with real heldout outputs and 48/48 Codex visual review. Mean mask/boundary/outside PSNR deltas were negative for all recipes, visual better rows were 0, and 30-step confirmatory micro remains forbidden. |
| Exp36 MiniMax objective rescue readback | `EXP36_READBACK_COMPLETED` | New isolated Exp36 branch/worktree created from Exp35 HEAD `fb70266`. Reread Exp30/Exp35 PRDs, registries, reports, metrics, and code. Previous MiniMax failures are no-change/slight-degradation, not collapse; inference uses weights and MiniMax is trainable, but no recipe is quality-positive. No GPU task, training, 30-step, RC-FPO, or protected-lane action launched. |
| Exp36 MiniMax no-change forensic audit | `MINIMAX_NOCHANGE_CAUSE_UTILITY_SCALE_TOO_WEAK` | Re-audited prior Exp30/Exp35 checkpoint, output-diff, loss-scale, winner-SFT, and rescue data with no new training. Checkpoint fallback and ignored trainable scope are not supported; previous objectives either produced near-constant utility or harmful heldout movement. 30-step remains locked. |
| Exp36 MiniMax inference sensitivity | `MINIMAX_INFERENCE_SENSITIVITY_PASS` | No-training PAI GPU0 diagnostic on 2 heldout + 2 train rows. Step0 identity replay max MAE `0.0`; temporary 1.01x transformer perturbation produced mean full/mask MAE `0.088218` / `0.156302`. Codex reviewed 4/4 strips: subtle nonzero response, no collapse/new artifact. This confirms inference uses weights, not quality-positive improvement. |
| Exp36 MiniMax trainable-scope audit | `MINIMAX_TRAINABLE_SCOPE_EXPANDED_S1_READY` | No-training scope contract added. S0 records previous full-transformer scope; S1 LoRA attention/projection rank8 alpha16 is ready for winner-SFT positive-control; S2 last-four-block MLP LoRA remains locked until S1 evidence. No GPU task, positive quality claim, 30-step, or RC-FPO launched. |

| Exp36 MiniMax winner-SFT positive-control | `MINIMAX_POSITIVE_CONTROL_PASS_HELDOUT_QUALITY_NOT_POSITIVE` | S0/S1 10-step winner-SFT reduced train loss and moved outputs, but heldout quality did not improve: Codex reviewed 24/24 strips, visual better 0, tie/no visible gain 20, and S0 high-LR new artifacts 4. | Bad-noise miner, objective rescue, 30-step, and MiniMax third-backbone-positive language remain locked. |

| Exp36 MiniMax final positioning | `MINIMAX_PLUMBING_POSITIVE_RECIPE_NOT_READY` / `TWO_BACKBONE_PLUS_MINIMAX_PLUMBING_ONLY` | MiniMax inference uses trained weights and winner-SFT moves outputs, but no heldout visual quality gain was found: Exp36 winner-SFT visual better 0/24 and previous Exp30/Exp35 preference recipes better 0. | Stop MiniMax objective rescue/30-step; keep paper language to DiffuEraser + VideoPainter confirmed, MiniMax plumbing-only. |

| Exp37 MiniMax LocalDPO-badnoise readback | `EXP37_READBACK_COMPLETED` | Readback from Exp30/35/36 rules out checkpoint/load and ignored-weight failures, but confirms MiniMax heldout quality remains non-positive across Exp30 0/32, Exp35 0/48, and Exp36 0/24 visual-better rows. | Next allowed step is train-vs-heldout diagnosis, then cleaner LocalDPO-style corruption and bad-noise scan; no 30-step, 2000-step, RC-FPO, or universal-adapter claim. |
| Exp37 MiniMax train-vs-heldout diagnosis | `MINIMAX_OBJECTIVE_SIGNAL_TOO_WEAK` | Exp36 S1 checkpoint-10 was evaluated on locked Gate64 train16/heldout16. Train local metrics were not meaningfully positive, heldout metrics were negative, and Codex reviewed 32/32 strips with 0 visual better rows. | Proceed only to cleaner LocalDPO-style local corruption and bad-noise diagnostics; 30-step, long training, and universal-adapter claims remain locked. |
| Exp37 LocalDPO-style OR corruption pool | `LOCALDPO_STYLE_POOL_READY_VISUAL_REVIEW_PASS` | Built train32/heldout16 local-corruption manifests from VOR-Train Gate64 rows only. Codex reviewed 48/48 selected sheets; final usable = 48/48 with 38 medium-hard and 10 hard-plausible, preserving auto-vs-final classifications. | Unlocks diagnostic bad-noise scan only; no 10-step rescue before preregistered states/recipes, and no 30-step or long training. |
| Exp37 MiniMax bad-noise diagnostic scan | `MINIMAX_BAD_NOISE_STATES_READY` | Scanned LocalDPO-style train32 with 64 states per row, total 2048 frozen-model candidate states, and wrote `hard_state_A/B/C` selections. Manifest SHA256 `492210b2cd725faa348adcbafaf37bf82cc6790b4eb0607b9f758047d1c795d4`. | Unlocks objective recipe preregistration only; no training before preregistration and no 30-step/long-training/universal-adapter claim. |
| Exp37 MiniMax objective rescue preregistration | `MINIMAX_OBJECTIVE_RESCUE_RECIPES_PREREGISTERED` | Locked R1 LocalDPO-Linear-HardNoise, R2 conditional SDPO-safe Linear, and R3 LocalDPO-SFTWarmup-Linear with utility scale 18.0, LR 1e-5, winner anchor 0.05, and outside preservation 0.02. | Unlocks only bounded 10-step execution exactly as preregistered; 30-step remains locked unless 10-step is positive. |
| Exp37 MiniMax LocalDPO-badnoise 10-step rescue | `MINIMAX_LOCALDPO_BADNOISE_PARETO_MIXED` / `MINIMAX_PLUMBING_POSITIVE_RECIPE_NOT_READY` | R1 produced mixed numeric movement (`+0.161946` mask PSNR, `-0.049755` boundary PSNR), while R2/R3 degraded full/mask/boundary/outside PSNR. Codex reviewed 48/48 heldout strips; each recipe had only 1/16 visually better rows. | 10-step positive gate failed, so 30-step is not unlocked. MiniMax remains plumbing-positive only; no third-backbone-success, long training, or universal-adapter claim. |
| Exp38 MiniMax full adapter breakthrough | `EXP38_READBACK_COMPLETED` | Isolated Exp38 branch/worktree created from Exp37. Readback confirms MiniMax is not a checkpoint/load or ignored-weight failure, but remains quality-negative/plumbing-positive after Exp30 `0/32`, Exp35 `0/48`, Exp36 `0/24`, and Exp37 `1/16` visible heldout improvements per recipe. GPU0/GPU1 are physically free on PAI, GPU2-GPU7 are occupied and untouched; no GPU task or training launched. |
| Exp38 MiniMax failure taxonomy | `MINIMAX_FAILURE_TAXONOMY_BUILT` | Decision tree now separates code/load, inference sensitivity, trainable scope, LR/update scale, objective signal, bad-noise/timestep, data difficulty, LocalDPO corruption strength, generalization, and evaluation sensitivity. Code/load and ignored inference are mostly ruled out; objective signal is strongest current hypothesis. Next gate is train-overfit diagnosis; no GPU task or training launched. |
| Exp38 MiniMax train-overfit diagnosis | `MINIMAX_OBJECTIVE_SIGNAL_TOO_WEAK_WITH_LOCAL_DRIFT` | Used PAI GPU0/GPU1 only. Exp37 R1 LocalDPO-badnoise checkpoint-10 changed outputs but was not quality-positive due outside/global drift; Exp36 S1 winner-SFT checkpoint-10 remained near no-change. Next Exp38 step is LocalDPO v2 / bad-noise v2, not 30-step or long training. |
| Exp38 MiniMax LocalDPO v2 / bad-noise v2 / SFT-DPO rescue | `MINIMAX_SFT_DPO_RESCUE_10STEP_NEGATIVE` | Built filtered LocalDPO v2 train30/heldout13, mined bad-noise v2 states, and ran R1/R2/R3 bounded 10-step rescue on GPU1. R1 had mild full/mask PSNR gains but boundary/outside tradeoffs and no clear visual quality win; R2/R3 were negative. 30-step and long training remain locked; MiniMax stays plumbing-positive only. |
| Exp40 MiniMax PSNR-safe rescue readback | `EXP40_READBACK_COMPLETED` | New isolated Exp40 branch/worktree created from Exp38. GPU0/GPU1 reserved; stale old Exp30 GPU0 heartbeat PGID `1715134` was recorded and terminated, with no compute PID and no KILL. R1 positive signal audited: `+0.102167` full PSNR and `+0.117230` mask PSNR, but boundary/outside/visual gates failed. Next step is R1 sample-level diagnosis before any new training. |
| Exp40 R1 sample-level diagnosis | `MINIMAX_R1_SIGNAL_AUDITED` | Existing Exp38 R1 evidence confirms boundary/outside cost and fogging/over-erasure risk. Exp38 SFT/DPO R1 has heldout13 only; train-side context uses existing train-overfit Exp37 R1 outputs. Next step is larger LocalDPO v3 pool and PSNR-safe SFT; DPO remains blocked until SFT is search/shadow safe. |
| Exp40 LocalDPO v3 pool | `MINIMAX_LOCALDPO_V3_POOL_READY_MINIMUM` | Built VOR-Train-only train64/search24/shadow24 local-corruption pool with zero split scene overlap and all selected rows medium-hard eligible. Full train96/search32/shadow32 target was not reached, so later claims must carry the minimum-pool caveat. |
| Exp40 Step0 baseline | `MINIMAX_STEP0_BASELINE_ESTABLISHED` | Ran raw MiniMax Step0 baseline on train64/search24/shadow24 using GPU0/GPU1 only. Shadow full/mask/boundary/outside PSNR = `26.209732` / `21.645338` / `24.277694` / `29.577002`; Codex opened 42 review pages covering 112/112 rows. Baseline only; no training, no hard comp, no VOR-Eval, no MiniMax positive claim. |
| Exp40 PSNR-safe SFT grid | `MINIMAX_SFT_PSNRSAFE_NEGATIVE` | Ran 12 winner-SFT-only 30-step recipes on GPU0/GPU1, evaluating 288 raw search outputs. Best aggregate recipe `SFTmC_S0_lr3em05` still had negative full/mask/boundary/outside deltas `-1.816781` / `-1.634597` / `-1.899575` / `-2.624405`; high-LR recipes showed noisy/color collapse. Stop before 100-step, DPO-after-SFT, and 300/500 confirmation. |
| Exp42 PAI MiniMax successful-removal + bad-noise data | `EXP42_PAI_MINIMAX_DATA_READBACK_COMPLETED` | New isolated PAI branch/worktree created from Exp40. Readback confirmed MiniMax is protocol-audited, inference-sensitive, and trainable, but previous direct SFT/DPO recipes remain quality-negative. PAI GPU0/GPU1 were free and no cleanup was needed. Next step is official MiniMax successful-removal mining and success-vs-failure bad-noise data construction on VOR-Train-derived sources only; no GPU mining, short training, DPO, H20 action, or VOR-Eval use launched by readback. |
| Exp42 official MiniMax successful-removal mining | `MINIMAX_SUCCESSFUL_REMOVAL_POOL_WEAK` | Official MiniMax raw mining completed on 117 VOR-Train-derived sources x 4 seeds = 468 candidates. Technical-valid = 468/468; automatic row-level counts were 52 successful-removal and 80 medium-hard failure candidates. Codex opened all selected compact temporal sheets. Human review found real successful-removal signal but heavy seed/source clustering: 18 success scene groups, 29 failure scene groups, and only 7 success/failure overlap groups; 37/80 auto-failures were borderline/noisy. Bad-noise v3, Stage2 data, SFT, and DPO remain locked pending targeted same-source pairing. |
| Exp44 PAI MiniMax targeted same-source mining | `EXP44_TARGETED_READBACK_COMPLETED` | New Exp44 branch/worktree created from Exp42 to fix the same-source pairing bottleneck without training. Source-group plan locked 56 target groups: 7 existing overlap groups, 11 success-only groups needing failure mining, 22 failure-only groups needing success mining, and 16 fallback near-miss groups. No GPU mining, SFT/DPO, optimizer step, H20 action, VOR-Eval use, or hard-comp output launched. |
| Exp44 targeted source manifest | `EXP44_TARGETED_SOURCE_MANIFEST_READY` | A/B/C-only targeted source manifest is ready for PAI GPU0/GPU1 mining: 40 source rows, 452 deterministic candidate seeds, 0 missing rows, SHA256 `5147839e1e2d60e0ecc9c77a438a934918605b5fa550fa58d1e3291df7be168b`. Mining runner preserves official MiniMax raw protocol and does not train. |
| Exp44 targeted second-pass mining | `MINIMAX_TARGETED_MINING_COMPLETED` | Official MiniMax raw mining completed 452/452 candidates on PAI GPU0/GPU1. Auto labels: success 138, medium-hard failure 231, auto same-source capacity 26 across 13 groups; visual relabel remains mandatory. | Proceed to strict visual relabeling; do not build training pairs or run SFT/DPO from auto labels alone. |
| Exp44 strict visual relabeling | `MINIMAX_TARGETED_RELABEL_COMPLETED` | Codex opened 47/47 selected review pages and conservatively relabeled all 452 candidates: SUCCESS_CLEAN 33, SUCCESS_USABLE 92, FAILURE_MEDIUM_HARD 137, rejected/nonusable 190. Same-source precheck: 10 overlap groups, one-to-one capacity 18, capped combination precheck 40. | Unlocks only explicit same-source pair construction and formal >=24 pair gate. No training, optimizer, bad-noise v4, Stage2 handoff, or MiniMax quality-positive claim yet. |
