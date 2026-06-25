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
| Exp26 VideoPainter v2 Gate64 review | `GATE64_VIDEO_REVIEW_COMPLETE_POOL_NOT_DATA_READY` | Reviewed 56 generated outputs: 31 medium-hard, 16 hard-plausible, 1 too-close, 8 trivial-bad. 47 eligible rows require balanced manifest construction before any DPO micro-training. |
