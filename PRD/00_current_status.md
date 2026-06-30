## 2026-06-30 Exp50 VOID Weight Download Blocked

Status: `VOID_WEIGHT_DOWNLOAD_BLOCKED`.

Official HF downloads for `netflix/void-model` and `alibaba-pai/CogVideoX-Fun-V1.5-5b-InP` failed from PAI with `httpx.ConnectError: [Errno 101] Network is unreachable`. No fallback or mirror was used. Exp50 remains blocked before env/model/inference gates until exact assets are available.


## 2026-06-30 Exp50 VOID Permission Recovery

Status: `VOID_ASSET_PERMISSION_RECOVERED`.

The user-provided root-side minimal permission fix was verified from PAI as `hj`. All six EXP50 directories passed read/write/execute/write-probe checks. No chmod/chown/root script was run by Codex. Exp50 can resume Milestone B1 official VOID repo download/audit.


## 2026-06-30 Exp50 VOID Asset Download Blocked

Status: `VOID_ASSETS_BLOCKED`.

Milestone B pre-download checks found that the requested VOID asset/output roots under `/mnt/nas/hj/H20_Video_inpainting_DPO` are not writable by `hj`; passwordless sudo and root SSH are unavailable. No assets were downloaded to avoid unapproved fallback writes. Exp50 is blocked until directory ownership is corrected or a fallback root is approved.


## 2026-06-30 Exp50 VOID Adapter Feasibility Readback

Status: `EXP50_VOID_READBACK_COMPLETED_WITH_NAS_PERMISSION_CAVEAT`.

ROSE is paused. VOID is now the preferred third-adapter candidate because public sources expose code, Pass1/Pass2 checkpoints, data-generation code, quadmask format, and training scripts. No VOID downloads, inference, training, or optimizer steps were run in Milestone A. PAI active worktree uses `/home/hj/H20_Video_inpainting_DPO_exp50_void_adapter` because the requested NAS-linked worktree and asset roots are not writable by `hj` yet.


## 2026-06-18 Exp19b Exploratory 2000 DAVIS50

Current best remains:

```text
Exp11 boundary outer b0.75 S2
```

At user request, the previously gated-off Exp19b branch was run as an
exploratory longer-training check:

```text
Exp19b Stage2 flow adapter, continued from 500 to 2000 total adapter steps
DAVIS50 eval completed
```

Important label note: the DAVIS evaluator row still prints
`Exp19b_stage2_500`, but the eval script loaded the exploratory 2000 adapter:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/exp19b_boundary_flow_adapter_s2_2000_exploratory_from500_limit100/last_weights/flow_adapter.pt
```

DAVIS50 result:

| Method | PSNR | SSIM | LPIPS | Ewarp | strict mask PSNR | boundary PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT-48000 | 32.665330 | 0.971062 | 0.016222 | 7.214799 | 21.021880 | 26.194571 |
| Exp11 outer b0.75 S2 | 32.840213 | 0.971818 | 0.015339 | 7.181782 | 21.196763 | 26.441316 |
| Exp19b exploratory 2000 | 32.840122 | 0.971818 | 0.015340 | 7.181850 | 21.196671 | 26.441224 |

Decision:

```text
Do not continue Exp19b under this setup.
```

Reason: the longer exploratory run did not amplify the tiny DAVIS10 temporal
signal. It is essentially tied with Exp11 and slightly worse on PSNR, SSIM,
LPIPS, strict-mask PSNR, boundary PSNR, and Ewarp. Representative DAVIS50
contact sheets also look tied rather than better.

## 2026-06-18 Exp19 Boundary-Gated Flow-Adapter DPO

Current best remains:

```text
Exp11 boundary outer b0.75 S2
```

Exp19 was scaffolded as an isolated flow-adapter experiment:

```text
exp19_boundary_gated_flow_adapter_dpo/
experiment_registry/exp19_boundary_gated_flow_adapter_dpo/
PRD/40_exp19_boundary_gated_flow_adapter_dpo.md
```

Status:

```text
DAVIS10_EVAL_COMPLETED_NEGATIVE_GATE
```

What changed:

- Recovered the architecture block with an isolated hook-based Stage2 wrapper
  under `exp19_boundary_gated_flow_adapter_dpo/`.
- The unsafe `additional_residuals` interfaces are no longer used.
- Flow adapters inject at:
  - `mid_block.motion_modules.0`
  - `up_blocks.0.motion_modules.0`
  - `up_blocks.1.motion_modules.0`
- Zero-init / gradient preflight passed:
  - enabled adapter output equals frozen Exp11 output
  - `base_grad_norm = 0`
  - adapter gradient is non-zero
- Exp19b boundary-gated Stage2 adapter-only 500 steps completed on PAI.
- `checkpoint-250`, `checkpoint-500`, and `last_weights` were saved.
- Exp19 inference wrapper was implemented in the Exp19 folder.
- `flow_adapter.pt` strict-loaded with no missing/unexpected keys.
- DAVIS10 completed with real adapter inference and DAVIS flow context.

DAVIS10 result:

| Method | PSNR | SSIM | LPIPS | Ewarp | strict mask PSNR | boundary PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT-48000 | 29.6181 | 0.9620 | 0.02204 | 8.3724 | 18.3203 | 24.2735 |
| Exp11 outer b0.75 S2 | 29.8295 | 0.9633 | 0.02065 | 8.3307 | 18.5317 | 24.6577 |
| Exp19b Stage2-500 | 29.8291 | 0.9633 | 0.02065 | 8.3306 | 18.5313 | 24.6574 |

TC was not computed because PAI could not download the OpenCLIP dependency used
by the TC backend. Ewarp was computed with the local RAFT backend.

Decision:

```text
Do not expand Exp19 to 1000 steps, DAVIS50, full cache, or full training.
Current best remains Exp11 outer b0.75 S2.
```

Reason: Exp19b is visually safe but indistinguishable from Exp11, Ewarp improves
by only `0.000080` absolute, and PSNR / strict mask / boundary PSNR have tiny
regressions.

## 2026-06-18 Exp19-R0 / Exp19c Refinement

Current best remains:

```text
Exp11 boundary outer b0.75 S2
```

Status:

```text
EXP19C_DAVIS10_COMPLETED_NEGATIVE_GATE
```

Completed:

- Exp19-R0 inference parity repair: `disabled_vs_Exp11_MAE = 0.0`.
- residual scale / confidence exponent sweep: best non-degrading setting is
  `scale=0.5`, `confidence_exponent=2.0`.
- real/zero/shuffled/reversed flow causality: real-flow wins only by a tiny
  Ewarp margin (`-0.000124` on the R0 subset).
- Exp19c-light latent warp continuation from Exp19b-500:
  `lambda_warp = 0 / 0.005 / 0.010 / 0.020`, each 500 steps.
- DAVIS10 metric and visual judgement.

DAVIS10 key result:

| Method | PSNR | SSIM | LPIPS | Ewarp | strict mask PSNR | boundary PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exp11 outer b0.75 S2 | 29.829309 | 0.963257 | 0.02065550 | 8.330730 | 18.531525 | 24.657501 |
| Exp19b Stage2-500 | 29.829470 | 0.963257 | 0.02065455 | 8.330525 | 18.531685 | 24.657372 |
| Exp19c0 lambda=0 | 29.829031 | 0.963255 | 0.02065269 | 8.330644 | 18.531247 | 24.657456 |
| Exp19c3 lambda=0.020 | 29.829368 | 0.963257 | 0.02065228 | 8.330675 | 18.531584 | 24.657357 |

Decision:

```text
Do not start Exp19d.
Do not run DAVIS50.
Do not continue Exp19 to 1000 or 2000 steps.
```

Reason: lambda>0 warp variants do not beat the lambda=0 continuation control
on Ewarp, do not produce clear visual better cases, and the metric deltas are
orders of magnitude below the positive gate.

## 2026-06-18 Exp18 PAI Gate Result

Current best remains:

```text
Exp11 boundary outer b0.75 S2
```

Exp18 Multi-frame Propagation-Confidence Gated DPO has now completed the
requested PAI gate:

```text
limit=100 propagation cache
Exp18a Stage1-500
Exp18b Stage1-500
Exp18c oracle Stage1-500 diagnostic
DAVIS10 metric + visual sanity
```

Result:

```text
PAI_GATE_COMPLETED_NEGATIVE_ABLATION
```

DAVIS10 summary:

| Method | PSNR | SSIM | strict mask PSNR | boundary PSNR |
| --- | ---: | ---: | ---: | ---: |
| Exp11 boundary outer b0.75 S2 | 30.2413 | 0.9650 | 18.7114 | 24.8326 |
| Exp18a prop-only S1-500 | 30.1024 | 0.9650 | 18.5725 | 24.7090 |
| Exp18b prop+gen S1-500 | 29.6892 | 0.9609 | 18.1593 | 24.7152 |
| Exp18c oracle S1-500 | 29.7626 | 0.9632 | 18.2326 | 24.7991 |
| SFT-48000 baseline | 30.0126 | 0.9635 | 18.4827 | 24.4772 |

Decision:

```text
Do not expand Exp18 to Stage1 1000, full cache, Stage1 2000, or Stage2.
Exp18 is an exploratory / negative ablation unless the formulation changes.
```

Reason: non-oracle propagation confidence is sparse, and even the oracle
diagnostic does not beat Exp11. Visual review also found no clearly positive
Exp18-over-Exp11 case.

## 2026-06-17 Exp18 Multi-frame Propagation-Confidence Gated DPO

Current best remains:

```text
Exp11 boundary outer b0.75 S2
```

Paused directions:

- OR / object-removal benchmark
- BR / VideoPainter adapter direction
- adaptive normalization variants
- Exp16 full prior cache / full training
- Exp17 saturation-positive follow-up

New direction:

```text
Exp18 Multi-frame Propagation-Confidence Gated DPO
```

Motivation: Exp16 used real ProPainter prior cache but its confidence was a
GT-error training oracle and did not beat Exp11. Exp18 instead asks which masked
pixels can be reliably propagated from other frames, and gates the DPO/x0 losses
between propagation-preservation and generation regions.

Implemented artifacts:

```text
exp18_multiframe_propagation_gated_dpo/
experiment_registry/exp18_multiframe_propagation_gated_dpo/
PRD/39_exp18_multiframe_propagation_gated_dpo.md
reports/exp18_context_and_code_audit.md
reports/exp18_propagation_confidence_audit.md
reports/exp18_x0_latent_loss_implementation_audit.md
```

Previous execution status before the PAI gate:

```text
IMPLEMENTATION_READY_ON_HAL
PAI_RUN_BLOCKED_IN_THIS_SESSION_BY_MISSING_PAI_MOUNT_OR_SSH
```

Superseded by the 2026-06-18 PAI gate result above.

## 2026-06-17 Exp16 Prior-Confidence Gated DPO

Current mainline remains:

```text
Exp11 boundary outer b0.75 S2
```

Paused for now:

- OR / object-removal benchmark
- BR / VideoPainter adapter direction
- adaptive normalization new variants
- further Exp11 / Exp12 boundary tuning

New direction:

```text
Exp16 Prior-Confidence Gated DPO
```

Motivation: Exp11 outer b0.75 S2 proved that region-local / boundary-aware DPO
helps, but DPO still does not know where the ProPainter prior is reliable. Exp16
adds a prior-confidence gate so reliable prior regions are preserved, unreliable
regions are generated with GT/context preference, and the outer boundary remains
seam-constrained.

Current implementation status:

- folder: `exp16_prior_confidence_gated_dpo/`
- registry: `experiment_registry/exp16_prior_confidence_gated_dpo/`
- PRD: `PRD/36_exp16_prior_confidence_gated_dpo.md`
- context audit: `reports/exp16_prior_confidence_context_audit.md`
- confidence audit: `reports/exp16_prior_confidence_map_audit.md`
- x0 audit: `reports/exp16_x0_prior_loss_implementation_audit.md`

Current status:

```text
EXP16_STAGE1_500_LIMIT100_DAVIS10_SANITY_COMPLETED
```

What changed on 2026-06-17:

- Existing manifests did not expose verified ProPainter prior paths.
- A real ProPainter prior cache was generated for `limit=100`.
- Confidence maps were computed with GT-error confidence.
- Stage1 preflight passed with real prior frames, VAE latent targets, and
  reconstructed predicted latent x0.
- Stage1 500 on the limit=100 cache completed and saved diagnostics,
  `checkpoint-250`, `checkpoint-500`, and `last_weights`.
- A DAVIS10 visual/metric sanity eval has been completed using an Exp16
  DPO-S1 + SFT-S2 hybrid checkpoint.
- Confidence diagnostics were fixed with mass-based inside-mask fields.
- DAVIS10 result: Exp16 improves over SFT-48000 but does not exceed Exp11
  outer b0.75 S2 on the primary PSNR / strict-mask / boundary metrics.

Remaining guardrails:

- Stage2 is not wired for Exp16 and must not be launched.
- Full cache and full 2000+2000 training are not approved.
- No DAVIS50/YouTubeVOS100 full metric claim exists for Exp16 yet.
- DPO diagnostics show high `implicit_acc` and high `loser_dominant_ratio`, so
  this is implementation validation, not a final method result.

Current Exp16 decision:

```text
Do not launch full prior cache or Stage1 2000 yet.
If Exp16 continues, first adjust lambda_prior / lambda_gen / confidence alpha or
add a schedule, then rerun a small gate.
```

## 2026-06-17 Exp17 Saturation-Aware Positive DPO

Current next direction:

```text
Exp17 Saturation-Aware DPO-Positive Region Loss
```

Reason: Exp11 outer b0.75 S2 remains the best method, but dpo_diag still shows
DPO saturation / loser-dominant risk. Exp16 prior-confidence was an
implementation validation and did not beat Exp11, so the next attempt focuses on
the DPO objective itself instead of adding more prior losses.

Exp17 variants:

- Exp17a: stronger DPOP-style positive preservation.
- Exp17b: saturation-aware margin DPO.
- Exp17c: combined positive + saturation.

First gate:

```text
Stage1 1000 for each variant + DAVIS10 visual/metric sanity.
No Stage2.
No VBench.
```

Registry:

```text
experiment_registry/exp17_saturation_positive_dpo/
```

Current status:

```text
COMPLETED_NEGATIVE_STAGE1_GATES
```

DAVIS10 result:

| Method | PSNR | SSIM | strict mask PSNR | boundary PSNR |
| --- | ---: | ---: | ---: | ---: |
| Exp11 boundary outer b0.75 S2 | 30.2950 | 0.9664 | 18.7651 | 24.7722 |
| SFT-48000 baseline | 29.6227 | 0.9616 | 18.0928 | 24.1247 |
| Exp17a positive S1-1000 | 29.7313 | 0.9632 | 18.2014 | 24.4509 |
| Exp17b saturation S1-1000 | 29.8542 | 0.9623 | 18.3243 | 24.4384 |
| Exp17c combined S1-1000 | 29.5117 | 0.9609 | 17.9818 | 24.4214 |

Decision:

```text
Exp17b is the best Exp17 variant, but no variant beats Exp11.
Do not run Exp17 Stage1 2000 or Stage2.
Current best remains Exp11 boundary outer b0.75 S2.
```

Reason:

- dpo_diag still shows high loser dominance.
- The saturation gate did not meaningfully trigger under the tested margin.
- Visual positives are isolated and not stable enough.

## 2026-06-15 Current Best / Evidence Status

Current best under the fixed DAVIS50 raw6 hard-comp protocol is:

```text
Exp11 boundary outer b0.75 S2
stage = DPO-S1 + DPO-S2
protocol = raw6, D+G off, no PCM, no mask dilation, no Gaussian blur, hard comp, frame-wise in-memory metric
```

Canonical DAVIS50 score:

```text
PSNR 33.013954
SSIM 0.972295
LPIPS 0.015363
VFID 0.175423
TC 0.971122
mask PSNR 24.167487
```

Selected visual evidence has been verified complete on HAL:

```text
/home/hj/dpo-2-1-exp/this_week_exp11_exp12/visual_evidence/exp11_outer_b075_s2_selected_visuals
```

The strongest positive case is `boat`. Usable positive cases are `rhino`,
`dog-agility`, `lucia`, and `blackswan`. `dance-jump` and `soccerball` are
caution/failure cases and should not be used as positive paper examples.

Exp11 outer b0.75 dpo diagnostics are present for Stage1 and Stage2 and copied
to the this-week archive. The diagnostic label is `LOSER_DOMINANT` for both
stages, but without the old raw-DPO winner-gap explosion. Report metrics and
visuals together with this residual diagnostic risk.

YouTubeVOS100 extension status:

- HAL source: `/home/hj/Video_inpainting_DPO/data/external/youtubevos_432_240`
- PAI target: `/mnt/workspace/hj/nas_hj/data/external/youtubevos_432_240_eval100`
- fixed-seed sample manifest: `/home/hj/dpo-2-1-exp/this_week_exp11_exp12/youtubevos100/sample_manifest.csv`
- PAI eval launcher: `scripts/run_exp11_outer_youtubevos100_framewise_protocol_pai.sh`
- Compare only SFT-48000 baseline vs Exp11 boundary outer b0.75 S2.
- Completed result: Exp11 improves over SFT-48000 on YouTubeVOS100
  (`PSNR 33.7238 vs 33.3968`, `SSIM 0.9711 vs 0.9701`,
  `LPIPS 0.0168 vs 0.0176`, `VFID 0.1925 vs 0.2007`).
- Final paper/PPT visual package:
  `/home/hj/dpo-2-1-exp/final_20_visual_cases_for_paper`.

Adapter feasibility status:

- Do not launch adapter training yet.
- First direct diffusion candidates: `VideoPainter`, then `FFF-VDI`.
- Output-level only candidates: `ProPainter`, `E2FGVI`, `STTN`.
- Frozen / related-work only for now: `MiniMax-Remover`, `CoCoCo`, `FloED`,
  `VACE`, `LGVI`, `RT-Remover`, `VideoComp`.

## 2026-06-09 Current Active Experiment State

The active experiment surface is now defined by
`experiment_registry/current_active.md`. Older exploratory Exp8/Exp9 gate files
that conflict with the current naming are moved, without deletion, to
`pending_delete/`.

Current completed / tracked experiment names:

```text
pre-Exp5 historical setup
Exp5
NewExp5
NewExp6
Exp7a-1
Exp7a-2
Exp8a-1
Exp8a-2
Exp8c-1
Exp8c-2
Exp9-1
Exp9-2
Exp10-1 partial
```

Current PAI target-domain status:

- Exp9 is complete.
  - Exp9-1 = Stage1 DPO + SFT Stage2 DAVIS validation.
  - Exp9-2 = Stage1 DPO + Stage2 DPO DAVIS validation.
  - Both have `dpo_diagnostics.csv`, DAVIS metrics, `index.html`, and
    four-column qualitative videos.
- Exp10-1 is not complete. The previous PAI run reached partial Stage1 and then
  repeated external `SIGTERM` events. A fresh no-resume/no-policy-init retry was
  also externally terminated:
  `RUN_VERSION=20260609_145145_exp10_fresh_d3n16_val24`, signal at
  `2026-06-09 14:55:32 CST`, `Signal 15 (SIGTERM) received by PID 2698917`.
  This rules out the old interrupted checkpoint as the primary cause.
- Exp11 remains blocked. Do not launch an Exp11 run that is only an Exp10 clone.
  Exp11 requires a passed train-time flow/prior consistency audit.

Frame policy:

- Existing D3 generated-loser training clips are 16-frame clips; keep
  `NFRAMES=16` unless D3 data is regenerated.
- DAVIS / ProPainter validation requires effective duration greater than 22
  frames; use `DAVIS_VIDEO_LENGTH=24`.
- Do not run DAVIS validation with 16 frames and do not fake 24-frame training by
  padding/repeating frames.

Execution policy:

- PAI is the current target for Exp10/Exp11 work, but Exp10 is currently blocked
  by repeated external SIGTERM even on fresh training.
- H20 should only be used when explicitly requested in the current instruction.
- Git-linked workflow remains: edit on HAL, push, then sync to PAI/H20 only when
  required for execution.

## 2026-06-06 Exp8a Full-Loss Regularized DPO DAVIS Result

Current PAI status is based on the user's pasted audit output. PAI remains manual-only for Codex: do not claim direct execution on PAI.

Exp8 has been split into two evidence units:

- `Exp8a`: D3 comp + ordinary full-loss regularized DPO. This baseline is now complete and negative on DAVIS.
- `Exp8b`: future region-weighted-loss ablation. Do not conflate it with the current run.

Current Exp8a definition:

```text
experiment = exp08_d3_comp_fullloss_wingap_lose025_s1s2_2000_davis_pai
data = D3 selected_primary_comp.repaired.pai_paths.jsonl
task = partial-mask video inpainting
weights = /mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000
prior = ProPainter prior required for DAVIS validation
loss_region_mode = full
stage = Stage1 2000 -> Stage1 DAVIS val -> Stage2 2000 -> Stage2 DAVIS val
metric = tools/run_inpainting_metric_eval.py / inference/metrics.py
VBench = not used
```

Exp8a true loss:

```text
m_w     = policy winner MSE
m_l     = policy loser MSE
m_w_ref = reference winner MSE
m_l_ref = reference loser MSE
win_gap  = m_w - m_w_ref
lose_gap = m_l - m_l_ref

L_total =
    -logσ{-0.5 * 10 * (win_gap - 0.25 * lose_gap)}
    + 0.05 * m_w
    + ReLU(win_gap)
```

Observed status from the 2026-06-06 12:28 CST PAI audit:

- Stage1 training completed on PAI at 2000/2000.
- Stage1 run dir:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260605_142442_exp08_d3_comp_fullloss_wingap_lose025_s1_2000_davis_pai`
- Confirmed Stage1 artifacts: `checkpoint-2000`, `last_weights`, `dpo_diagnostics.csv`.
- Stage1 validation output:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp08a_fullloss_stage1_val_davis_20260606_070556`
- Stage2 training completed at 2000/2000.
- Stage2 run dir:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260606_070556_exp08_d3_comp_fullloss_wingap_lose025_s2_2000_davis_pai`
- Confirmed Stage2 artifacts: `checkpoint-2000`, `last_weights`, `dpo_diagnostics.csv`.
- Stage2 validation output:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp08a_fullloss_stage2_val_davis_20260606_070556`
- Latest successful continuation log:
  `logs/pipelines/exp08_d3_comp_fullloss_continue_after_s1_fixsafety_len24_writerpatched_pai_20260606_070556.log`
- PAI audit report:
  `reports/pai_exp8a_safe_audit_py_20260606_122809.md`

DAVIS metrics:

```text
Stage1 val: DPO-S1 + SFT-S2 vs DiffuEraser-base
  boundary_psnr_mean: 16.1306 vs 23.1742
  boundary_ssim_mean: 0.4964 vs 0.7861
  mask_region_psnr_mean: 15.6757 vs 22.7633
  mask_region_ssim_mean: 0.4813 vs 0.7754
  whole_video_psnr_mean: 23.9554 vs 29.4647
  whole_video_ssim_mean: 0.9017 vs 0.9564

Stage2 val: DPO-S1 + DPO-S2 vs DiffuEraser-base
  boundary_psnr_mean: 15.7133 vs 23.0682
  boundary_ssim_mean: 0.4790 vs 0.7804
  mask_region_psnr_mean: 15.2577 vs 22.6570
  mask_region_ssim_mean: 0.4638 vs 0.7695
  whole_video_psnr_mean: 23.5677 vs 29.3802
  whole_video_ssim_mean: 0.8967 vs 0.9558
```

DPO diagnostic readout:

```text
Stage1 final row, global_step=2000:
  dpo_loss=0.164854
  implicit_acc=1.000000
  loser_dominant_ratio=1.000000
  mse_l_over_ref_mse_l=300.585236
  sigma_term=0.869146
  kl_divergence=0.379077

Stage2 final row, global_step=2000:
  dpo_loss=0.595376
  implicit_acc=1.000000
  loser_dominant_ratio=1.000000
  mse_l_over_ref_mse_l=51.705112
  sigma_term=0.561856
  kl_divergence=0.050694
```

Conclusion:

- Exp8a is complete and negative.
- It must not be reported as a region-loss result.
- It must not be reported as a success: both Stage1 and Stage2 are much worse
  than DiffuEraser-base on DAVIS boundary, mask-region, and whole-video metrics.
- DPO diagnostics show a loser-degradation shortcut: the objective is largely
  satisfied by increasing loser error relative to the reference rather than by
  improving the winner.
- Stage2 DPO does not rescue the run. It is slightly worse than Stage1 on DAVIS
  metrics.

Fixes already applied on PAI during continuation:

- Added missing SD1.5 `feature_extractor/preprocessor_config.json`.
- Disabled missing SD1.5 safety checker in `diffueraser/diffueraser.py` by passing `safety_checker=None`, `feature_extractor=None`, and `requires_safety_checker=False`.
- Relaunched DAVIS validation with `DAVIS_VIDEO_LENGTH=24` after the 16-frame run failed because the effective duration was below the DiffuEraser/ProPainter minimum.
- Removed `macro_block_size=1` from the Exp8a visualization writer after PyAV
  rejected that keyword.

Do not rerun Exp8a. Use it as negative evidence for the full-loss D3 comp
baseline and compare future Exp8b/Exp8c only after those runs have their own
DAVIS metrics and dpo summaries.

## 2026-06-05 Exp8 DAVIS Region-Loss Diagnostic Prepared

Current execution boundary:

- H20 is generating new Exp7-fix small-mask D2 data. Do not stop H20, kill jobs, start H20 training, or take H20 GPUs.
- PAI is manual-only from Codex. The agent may prepare scripts and a copy/paste command block, but must not claim PAI execution.

Prepared next PAI diagnostic:

```text
experiment = exp08_d3_comp_regionloss_wingap_lose025_s1s2_2000_davis_pai
registry = experiment_registry/exp08_d3_comp_regionloss_davis_stage1stage2_2000
launcher = scripts/launch_exp8_d3_comp_regionloss_s1s2_2000_davis_pai.sh
data = D3 selected_primary_comp.repaired.pai_paths.jsonl
task = partial-mask video inpainting
weights = /mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000
prior = ProPainter prior required for DAVIS inference/validation
loss_region_mode = region
stage = Stage1 2000 -> DAVIS val -> Stage2 2000 -> DAVIS val
metric = tools/run_inpainting_metric_eval.py using inference/metrics.py
VBench = not used
```

Exp8 true loss:

```text
m_w, m_l, m_w_ref, m_l_ref are region-weighted MSE values.
win_gap = m_w - m_w_ref
lose_gap = m_l - m_l_ref

L_total =
    -logσ{-0.5 * 10 * (win_gap - 0.25 * lose_gap)}
    + 0.05 * m_w
    + ReLU(win_gap)

region weights: mask=1.0, boundary=0.5, outside=0.05
```

The launcher must stop before training if the D3 PAI manifest, SFT-48000 weights, ProPainter weights, DAVIS data, region-loss implementation, or dpo diagnostics support is missing.

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

# Current Status

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


Updated: 2026-06-02

## 2026-06-04 CST Exp9 Clean Gate / Target Eval Update

PAI clean Exp9-comp gate:

```text
experiment = exp9_youtubevos_d3_partialmask_wingap_lose025_stage1_gate1500
status = clean gate running
manual_launch_report = reports/pai_exp9_comp_clean_launch_report.md
manifest = selected_primary_comp.repaired.pai_paths.jsonl
config_confirmed = Stage1 only, partial mask, mask_from_manifest=true, beta=10,
                   winner_gap=1.0, lose_gap=0.25, max_steps=1500,
                   ckpt_steps=500, ckpt_limit=5
invalid_old_run = do not evaluate
```

H20 Exp9-nocomp gate:

```text
experiment = exp9_youtubevos_d3_nocomp_partialmask_wingap_lose025_stage1_gate1500_h20
status = Stage1 gate completed
checkpoints = checkpoint-500, checkpoint-1000, checkpoint-1500, last_weights
directly_evaluable_now = last_weights
note = checkpoint-500/1000/1500 are accelerator state dirs and need export
       before direct inference eval
```

H20 target-domain eval status:

```text
youtubevos_d3_nocomp_eval = completed on H20 using D3 selected-primary-nocomp
eval_log = logs/pipelines/exp9_d3_nocomp_target_eval_20260604_023243.log
eval_output = logs/target_eval/exp9_d3_nocomp_gate_h20_20260604_023243
samples = 200 generated videos, 30 side-by-side videos, 100 metric samples
metric_summary = Exp9_nocomp_last improves mask PSNR/SSIM over base but worsens
                 whole-video SSIM, boundary PSNR, outside max diff, and temporal
                 delta; do not continue nocomp long training before comp eval.
davis_eval = blocked until target prediction generation and pair_manifest are prepared
metric_policy = use inference/metrics.py through project wrappers; no VBench
```

## 2026-06-03 Exp9 Gate Monitoring Boundary

PAI and H20 execution boundaries are explicit:

- PAI is manual-only from Codex. Provide copy/paste shell blocks and wait for
  pasted output before judging or stopping a run.
- H20 can be inspected by SSH from HAL. H20 monitoring reports should be
  written directly under `/home/nvme01/H20_Video_inpainting_DPO/reports/`.

Current Exp9 target-domain gates:

| Host | Experiment | Status | Manifest | Max steps | Action |
| --- | --- | --- | --- | ---: | --- |
| PAI | `exp9_youtubevos_d3_partialmask_wingap_lose025_stage1_gate1500` | stopped; invalid as Exp9 gate | D3 selected-primary-comp PAI paths | actual log showed 10000 | do not use as Exp9 gate; stale env wrote old Exp5 run name |
| H20 | `exp9_youtubevos_d3_nocomp_partialmask_wingap_lose025_stage1_gate1500_h20` | finished normally | D3 selected-primary-nocomp H20 paths | 1500 confirmed | ready for target-domain eval |

H20 monitor snapshot, 2026-06-03 14:21 CST:

```text
report = /home/nvme01/H20_Video_inpainting_DPO/reports/h20_exp9_nocomp_gate_monitor_report.md
status = running
current_step = about 190 / 1500
max_steps_detected = 1500
stage_policy = Stage1 only; no DPO Stage2; no VBench
gpu_policy = GPU 0-5 only; GPU 6/7 idle in the initial launch check
checkpoint_status = no checkpoint yet; wait for checkpoint-500 / 1000 / 1500
dpo_diagnostics_csv = present
errors = no Traceback / OOM / SIGFPE detected in monitor scan
```

PAI monitor / stop update, 2026-06-03 15:03 CST:

```text
report = reports/pai_exp9_comp_gate_stop_report.md
status = manually stopped
evidence = global_step about 4850 and progress 4856 / 10000
stopped_pid = 3400017
final_process_check = empty for exp9/train_stage1/accelerate/lingbot-worldphy matcher
checkpoint_status = saved under stale output dir, not an Exp9-named gate dir
stale_output_dir = /mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260603_065327_exp5_d2_comp_k4_stage2_full
available_checkpoints = checkpoint-2000, checkpoint-4000
interpretation = invalid / contaminated by stale env; not comparable to gate1500
```

Updated H20 monitor snapshot, 2026-06-04 01:08 CST:

```text
status = finished normally
current_step = 1500 / 1500
max_steps_detected = 1500
gpu_policy = all GPUs idle after completion
checkpoint_status = checkpoint-500, checkpoint-1000, checkpoint-1500, last_weights
errors = no Traceback / OOM / SIGFPE detected
```

Next after both gates have evaluable checkpoints:

- Run target-domain video inpainting evaluation on YouTube-VOS / DAVIS.
- Use `tools/run_inpainting_metric_eval.py`, which calls
  `inference/metrics.py`.
- Do not use VBench for Exp9 inpainting evaluation.
- Do not train DPO Stage2, start Exp8, regenerate D2/D3, or launch a long run
  before the comp-vs-nocomp gate report is reviewed.

## D2 Data Readiness

The D2 generated-loser dataset is ready and must not be regenerated for the
beta10 reruns.

Root:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4
```

Ready manifests:

| Manifest | Rows |
| --- | ---: |
| `selected_primary_comp.repaired.jsonl` | 10000 |
| `selected_primary_nocomp.repaired.jsonl` | 10000 |
| `selected_secondary_comp.repaired.jsonl` | 10000 |
| `selected_secondary_nocomp.repaired.jsonl` | 10000 |

## Exp5 Collapse Status

`exp5_d2_comp_k4_stage1/stage2_full` with `beta_dpo=500` and 10000-step
Stage1/Stage2 is marked **failed / collapsed / diagnostic only**.

Evidence:

- Stage2 10000 full-mask VBench qualitative outputs show visual collapse.
- Side-by-side videos show the exp5 side as high-frequency noise and color
  explosion rather than coherent inpainting.
- DPO diagnostics saturated early with `acc=1`, `dpo=0`, and `loss=0`;
  this indicates preference-objective saturation, not image quality.
- VBench showed weak downstream behavior, including `dynamic_degree=0`,
  low `overall_consistency`, and poor scene/spatial/object dimensions.

Interpretation:

- This is not a task or code-path failure.
- Exp3 showed the VideoDPO-to-DiffuEraser DPO bridge can work.
- Exp5 failed because D2 generated losers plus full-mask training, full-video
  loss, `beta_dpo=500`, no SFT regularization, and long 10000-step training
  over-optimized the preference signal and pushed the model off distribution.
- Old Exp5 beta500 must not be used as a final result. Keep it only as a
  failed ablation and diagnostic artifact.

`exp5_d2_comp_k4_beta10_s1s2_4000` is also marked **failed / collapsed /
diagnostic only**. Its Stage2 correctly loaded Stage1 `last_weights` and
completed 4000 steps, but the visual outputs collapsed into prompt-insensitive
stripe/high-frequency textures. The log shows the same degenerate mechanism:

- `mse_w >> ref_mse_w`: the policy got worse on the winner.
- `mse_l >> ref_mse_l`: the policy got even worse on the loser.
- `win_gap` and `lose_gap` both grew, while `implicit_acc` stayed near 1.
- `dpo_loss` often approached 0, so the ranking objective was satisfied even
  though visual quality collapsed.

This confirms the failure is not a Stage2 handoff bug and not a VBench script
bug. It is a degenerate solution of D2 generated-loser + full-mask/full-video
DPO: ranking can be improved by making the loser worse while also damaging the
winner.

Replacement:

| Experiment | Status | beta_dpo | Stage1 | Stage2 | Eval |
| --- | --- | ---: | ---: | ---: | --- |
| `exp5_d2_comp_k4_wingap_lose025_beta10_s1s2_4000` | planned/running | 10 | 4000 | 4000 | qual30 + full VBench |

## H20 Old Exp6 beta500 Status

All old H20 Exp6 training using the unanchored objective is superseded and
should be stopped if still running.

Reason:

- It has the same collapse risk exposed by Exp5 beta500 and Exp5 beta10.
- It should not be continued as a final result.

Replacement:

| Experiment | Status | beta_dpo | Stage1 | Stage2 | Eval |
| --- | --- | ---: | ---: | ---: | --- |
| `exp6_d2_nocomp_k4_wingap_lose025_beta10_s1s2_4000` | planned/running | 10 | 4000 | 4000 | qual30 + full VBench |

## Current Run Policy

- Do not continue old beta500 Exp5/Exp6 or unanchored beta10 Exp5/Exp6 long
  training.
- Do not regenerate D2.
- Do not restore D1 full-mask work.
- Do not touch Exp8 region-loss settings in this pass.
- New reruns use winner-anchored DPO with `beta_dpo=10`,
  `lose_gap_weight=0.25`, `winner_abs_reg_weight=0.05`,
  `winner_gap_reg_weight=1.0`, 4000 Stage1 steps, 4000 Stage2 steps, no
  validation during training, then automatic qual30 and full VBench.

## 2026-05-31 Exp7 Partial-Mask Gate

`exp5_d2_comp_k4_wingap_lose025_beta10_s1s2_4000` is improved relative to the
unanchored Exp5 runs: winner-gap regularization suppresses the universal blue
stripe/high-frequency collapse mode. It is not yet a final success because the
data-only full-mask/full-video objective still shows texture and color
attractors in qualitative side-by-side videos.

H20 `exp6_d2_nocomp_k4_wingap_lose025_beta10_s1s2_4000` is running and must
continue. Action: monitor only; do not kill. Purpose: no-comp data-only
comparison against Exp5 comp.

2026-06-02 H20 check:

```text
status = running
current_stage = Stage2
stage1_completed = 2026-06-01 22:51 CST
stage2_progress = about 420 / 4000 steps
num_gpus = 6
gpu_policy = using GPU 0-5 only; GPU 6/7 idle
stage2_handoff = loaded Stage1 last_weights
qual30 = not started
full_vbench = not started
```

Current H20 diagnostics remain mixed: winner anchoring keeps `win_gap` near
zero and `mse_w_over_ref_mse_w` close to 1, but `loser_dominant_ratio=1.0`,
`mse_l_over_ref_mse_l` is high, and `sigma_term` can approach saturation. This
is still a valid no-comp comparison run; continue monitoring and do not stop it
unless explicitly requested.

Next gate:

```text
EXP_NAME = exp7_d2_comp_k4_partial_wingap_lose025_beta10_s1s2_gate1500
status = planned / launching on PAI
purpose = test whether aligning the training task with D2 partial-mask loser
          generation stabilizes training
manifest = selected_primary_comp.repaired.jsonl
train_mask_mode = partial
mask_from_manifest = true
M_train = M_gen from manifest mask_path
loss_region_mode = full
beta_dpo = 10
winner_abs_reg_weight = 0.05
winner_gap_reg_weight = 1.0
winner_gap_reg_margin = 0.0
lose_gap_weight = 0.25
stage1_steps = 1500
stage2_steps = 1500
validation = qual30 side-by-side + dpo_diag summary
full_vbench = disabled by default for gate
```

Exp5/6 remain data-only full-mask bridge experiments. Exp7 is a task-alignment
gate: DiffuEraser receives the same partial mask used to generate the D2 loser.
If Exp7 is more stable than Exp5, D2 is not inherently bad; the main issue is
the data-only full-mask objective mismatch.

## 2026-06-01 Exp7 Gate1500 Evaluation Status

`exp7_d2_comp_k4_partial_wingap_lose025_beta10_s1s2_gate1500` completed the
planned gate training:

```text
train_mask_mode = partial
mask_from_manifest = true
beta_dpo = 10
winner_abs_reg_weight = 0.05
winner_gap_reg_weight = 1.0
lose_gap_weight = 0.25
stage1_steps = 1500
stage2_steps = 1500
```

Observed with the current full-mask-style qual30:

- Full-mask qual30 looks poor and stripe-heavy.
- Some samples are worse than the new Exp5 winner-anchored full-mask run.
- Winner-gap regularization keeps `win_gap` relatively bounded.
- Loser degradation remains strong: `loser_dominant_ratio` reaches 1.0 and
  `mse_l_over_ref_mse_l` can become very high.

Interpretation:

- Do not mark Exp7 as success.
- Do not mark Exp7 as final failure yet.
- The current qual30 is task-mismatched because Exp7 trains partial-mask
  inpainting (`M_train = M_gen` from manifest `mask_path`), while the qual30
  path is still full-mask prompt generation.
- Current status: **inconclusive / risky**.
- Record full-mask qual30 as **failed / task-mismatched** and run a true
  partial-mask manifest evaluation before deciding whether Exp7 failed.

Completed task-matched eval:

```text
scripts/eval_exp7_partialmask_gate.sh
```

This uses D2 `win_video_path` and `mask_path` to compare DiffuEraser-base and
Exp7 checkpoints on the actual partial-mask inpainting task. Do not launch
full Exp7 4000+4000, full VBench, or Exp8 before this report is reviewed.

## 2026-06-02 Exp7-PM-Gate1500 Partial-Mask Eval

Name:

```text
Exp7-PM-Gate1500 = D2-comp partial-mask task-alignment gate
```

Output:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/partialmask_eval/exp7_gate1500_20260602_000500
```

Artifacts:

```text
side_by_side videos = 60
index.html = present
pair_manifest.csv = present
metrics/summary.csv = present
report.md = present
```

The PAI eval initially failed while reading generated mp4 files because
`imageio` selected a `pyav` backend incompatible with the installed `av`
version (`ContainerFormat.variable_fps` missing). The checked-in eval tool now
uses ffmpeg rawvideo decoding for input videos.

Result:

| Model | mask_region_psnr_mean | mask_region_ssim_mean | outside_region_diff_mean_mean | temporal_diff_delta_vs_gt_mean |
| --- | ---: | ---: | ---: | ---: |
| DiffuEraser-base | 8.99765 | 0.272146 | 2.91477 | 5.58378 |
| Stage1_last | 9.57079 | 0.288404 | 2.92006 | 12.7824 |
| Stage2_last | 7.88448 | 0.235938 | 2.91600 | 6.52143 |

Interpretation:

- Full-mask qual30 remains failed / task-mismatched for Exp7.
- True partial-mask eval shows `Stage1_last` beats DiffuEraser-base on
  mask-region PSNR and SSIM.
- `Stage2_last` regresses below both `Stage1_last` and DiffuEraser-base.
- Stage1 checkpoint-500 and checkpoint-1000 were not available in the exported
  run directory, so only Stage1 last and Stage2 last were evaluated.
- Exp7 is no longer a total failure: partial-mask task alignment is promising.
- Do not launch full Exp7 4000+4000 yet; Stage2 regression must be addressed.
- The prepared no-lose-gap gate is the next likely diagnostic if visual review
  confirms loser-degradation artifacts.

## 2026-06-02 Critical Stage Interpretation Correction

DiffuEraser is a two-stage model:

- Stage1 mainly controls spatial generation quality, BrushNet, UNet2D, and
  appearance.
- Stage2 mainly controls video temporal consistency, motion module behavior,
  and temporal modeling.

Therefore the Exp7-PM-Gate1500 result must not be interpreted as "final
inference should use Stage1 only." The correct candidate is:

```text
DPO-DiffuEraser-Stage1 spatial / appearance weights
+
frozen SFT-DiffuEraser-Stage2 temporal / motion weights
```

Current conclusion:

- Exp7 Stage1_last improves true partial-mask metrics over DiffuEraser-base.
- Exp7 Stage2_last regresses below both Stage1_last and DiffuEraser-base.
- DPO Stage2 is currently harmful and should remain stopped.
- DPO should target Stage1 spatial quality first.
- The temporal / motion prior should be preserved from a validated SFT Stage2
  checkpoint, ideally the best YouTube-VOS SFT Stage2 if found.

Important checkpoint rule:

- Do not simply load a complete SFT Stage2 checkpoint over the DPO result.
- A complete Stage2 checkpoint may contain both spatial and temporal weights.
- The hybrid must keep DPO Stage1 spatial/appearance weights and only preserve
  SFT Stage2 temporal/motion weights.

New first-priority audit/eval:

```text
name = exp7_pm_dpoS1_sftS2_hybrid_ckptsweep
task = true partial-mask manifest eval
candidate = DPO Stage1 checkpoint X + frozen SFT Stage2 motion checkpoint Y
full_vbench = disabled
training = none
```

Prepared tooling:

- `tools/inspect_diffueraser_stage_weights.py`
- `tools/build_diffueraser_dpoS1_sftS2_hybrid.py`
- `scripts/eval_exp7_dpoS1_sftS2_hybrid_partialmask.sh`
- `scripts/launch_exp7_pm_stage1only_ckptsweep_pai.sh` prepared only; do not
  run automatically.
## 2026-06-02 Mainline Correction: Target Domain

The final evaluation and demo domains are **YouTube-VOS** and **DAVIS**.
VideoDPO is a bridge domain only: it is useful for native VideoDPO repo
integration, DiffuEraser adapter validation, generated-loser manifest plumbing,
partial-mask support, DPO diagnostics, and Stage1/Stage2 loading ablations. It
is not the final target domain.

Do not continue planning VideoDPO partial-mask SFT warmup. If SFT is needed, it
should be target-domain SFT or related video-inpainting data, not VideoDPO
bridge warmup.

Current priority:

1. Run target-domain eval on YouTube-VOS / DAVIS using existing checkpoints and
   stage compositions.
2. Sync D3 YouTube-VOS generated-loser data to PAI only as background data
   preparation.
3. After D3 sync, run D3 post-generation audit, path rewrite, and readiness.
4. Enter Exp9 target-domain DPO only if target-domain eval shows that
   VideoDPO-bridge DPO does not transfer.

H20 D3 audit:

- root: `/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4`
- report: `/home/nvme01/H20_Video_inpainting_DPO/reports/d3_h20_audit_report.md`
- size/files: 249G / 1,819,879 files
- selected primary rows: 3,327 comp and 3,327 no-comp
- sampled 100 rows: status OK, 16 frames, 512x320, readable
- all sampled paths are H20-only `/home/nvme01/...`; PAI path rewrite is required before training.

## 2026-06-03 Target-Domain Mainline

Current mainline:

- VideoDPO is a bridge domain only. It validates model replacement, native
  VideoDPO integration, generated-loser manifests, partial-mask plumbing,
  winner-gap regularized DPO, diagnostics, and Stage1/Stage2 loading.
- YouTube-VOS and DAVIS are the final target domains for evaluation and
  reporting.
- Do not start Exp8, VideoDPO warmup, DPO Stage2, full VBench, or any long
  4000+4000 run in this phase.

Metric policy:

| Task type | Metric backend |
| --- | --- |
| video generation / full-mask VBench prompt generation | VBench |
| video inpainting / partial-mask inpainting | project metric module (`inference/metrics.py`; no standalone `metric.py` exists in this checkout) |

The target-domain YouTube-VOS / DAVIS partial-mask eval must use the existing
project metric functions through `tools/run_inpainting_metric_eval.py`. Do not
reimplement PSNR, SSIM, mask-region metrics, boundary metrics, or temporal
metrics in new scripts. New wrappers may only organize inputs, call the
existing metric module, and summarize results.

D3 status:

- D3 is the YouTube-VOS generated-loser data asset for Exp9.
- PAI slim sync completed for selected-primary data.
- D3 full readiness = false under the current slim sync because secondary
  manifests are absent.
- D3 primary-comp gate readiness = true once the selected-primary-comp repaired
  manifest has clean PAI paths and the primary-comp sample audit has zero
  issues. Secondary absence must not block the first Exp9 Stage1 gate.
- `selected_primary_comp.repaired.pai_paths.jsonl` is the first Exp9 gate
  training manifest if readiness checks pass on PAI.
- Slim sync does not make candidates/secondary manifests fully ready; do not
  run workflows that require full D3 candidates or secondary selections unless
  they are synced separately.

Exp9 boundary:

- Exp9 is target-domain partial-mask DPO.
- First run is Stage1-only gate:
  `exp9_youtubevos_d3_partialmask_wingap_lose025_stage1_gate1500`.
- Train with `train_mask_mode=partial`, `mask_from_manifest=true`,
  `loss_region_mode=full`, winner anchoring, `beta_dpo=10`,
  `lose_gap_weight=0.25`, `winner_abs_reg_weight=0.05`,
  `winner_gap_reg_weight=1.0`, and `sft_reg_weight=0.0`.
- Do not train DPO Stage2. Eval uses target-domain inpainting metrics, not
  VBench.

## 2026-06-03 H20 Complementary Work

Current split:

- PAI: Exp9 D3-comp Stage1 gate.
- H20: Exp9 D3-nocomp Stage1 gate on GPUs 0-5.

Reason:

- New Exp6 no-comp full-mask qual30 has a qualitative hypothesis that longer
  prompts may look better than DiffuEraser-base.
- This observation is not a conclusion; it requires prompt-length stratified
  audit via `tools/analyze_new_exp6_prompt_length_effect.py`.
- Since PAI is running the comp target-domain gate, H20 should run the nocomp
  target-domain gate to form a comp-vs-nocomp comparison.

H20 nocomp gate:

- `exp9_youtubevos_d3_nocomp_partialmask_wingap_lose025_stage1_gate1500_h20`
- manifest: D3 `selected_primary_nocomp.repaired.jsonl` or readable H20-local
  `selected_primary_nocomp.jsonl`
- Stage1 only; no DPO Stage2
- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5`
- target-domain eval uses `inference/metrics.py`, not VBench

## 2026-06-06 CST H20 Exp8c GT-Win Target-Domain Diagnostic

Exp8c was added as a target-domain diagnostic after finding that the D3
generated-loser manifest's cached `win_video_path` can differ from the original
YouTube-VOS frame sequence for some rows. Exp8c keeps the D3 selected comp
loser and mask unchanged, but replaces the winner path with an aligned cache of
original YouTube-VOS GT frames selected by each candidate's
`canonical_frame_indices`.

Scope:

```text
experiment = exp08c_youtubevos_gtwin_d3comp_fullloss_wingap_lose025_s1s2_2000_davis_h20
machine = H20
worktree = /home/nvme01/H20_Video_inpainting_DPO_exp8c_gtwin
source commit = 9c0eba6683992b8a15765e47475085ee475939cf
manifest = /home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/exp08c_youtubevos_gtwin_d3comp_lose_fixed/manifests/selected_primary_comp.gtwin.jsonl
manifest rows = 3327
winner = original YouTube-VOS GT frames aligned by canonical_frame_indices
loser = unchanged D3 selected-primary-comp final_loser_video_path
mask = unchanged D3 selected-primary-comp mask_path
task = partial-mask video inpainting
loss_region_mode = full
Stage1 = 2000 steps
Stage2 = 2000 steps after Stage1 DAVIS validation
validation = DAVIS with ProPainter prior and inference/metrics.py wrapper
VBench = not used
```

Loss:

```text
win_gap = m_w - m_w_ref
lose_gap = m_l - m_l_ref

L_total =
  -logsigmoid(-0.5 * 10 * (win_gap - 0.25 * lose_gap))
  + 0.05 * m_w
  + ReLU(win_gap)
```

H20 precision fix:

The first Exp8c H20 launch used the default bf16/split-positive-negative path
and failed at step 0 with `SIGFPE`. This matches earlier PRD notes: H20 can hit
`SIGFPE` in bf16 paths, and `SPLIT_POS_NEG_FORWARD=1` is unstable under H20
DDP.

The validated H20-safe configuration is:

```text
MIXED_PRECISION = no
POLICY_DTYPE = fp32
VAE_DTYPE = fp32
REF_DTYPE = fp32
TEXT_DTYPE = fp32
SPLIT_POS_NEG_FORWARD = 0
CUDA_VISIBLE_DEVICES = 1,2,3,4,5,6,7
GPU0 = reserved / not used
```

Evidence:

```text
bf16/split first run:
  run_dir = /home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260606_105422_exp08c_youtubevos_gtwin_d3comp_fullloss_wingap_lose025_s1_2000_davis_h20
  result = failed at step 0 with SIGFPE

fp32/nosplit smoke:
  run_dir = /home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260606_113407_exp08c_youtubevos_gtwin_d3comp_fullloss_smoke_fp32_nosplit_gpu1_step1_h20
  result = passed 1 step and wrote dpo_diagnostics.csv

formal fp32/nosplit run:
  pid = 333723
  log = /home/nvme01/H20_Video_inpainting_DPO_exp8c_gtwin/logs/pipelines/exp08c_youtubevos_gtwin_d3comp_fullloss_wingap_lose025_s1s2_2000_davis_h20_fp32_nosplit_20260606_114413.log
  stage1_run_dir = /home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260606_114414_exp08c_youtubevos_gtwin_d3comp_fullloss_wingap_lose025_s1_2000_davis_h20
  status at 2026-06-06 12:36 CST = running, reached global_step=150, dpo_diagnostics.csv present, no checked errors
```

Current diagnostic snapshot from the formal run:

```text
global_step=150
implicit_acc=0.714286
dpo_loss=0.693146
winner_abs_reg=0.001859
winner_gap_reg=0.000053
mse_w=0.001859
ref_mse_w=0.001816
mse_l=0.001824
ref_mse_l=0.001649
win_gap=0.000044
lose_gap=0.000176
mse_w_over_ref_mse_w=1.024070
mse_l_over_ref_mse_l=1.106694
sigma_term=0.500000
kl_divergence=0.000055
```

Interpretation:

- Exp8c is not a final success claim; it is a controlled GT-winner diagnostic.
- It tests whether directly supervising the winner against original aligned
  YouTube-VOS GT reduces the winner-target mismatch seen in cached D3 winners.
- The `SIGFPE` fix should be treated as an H20 execution rule for DPO gates:
  use fp32 paths and disable split-positive-negative forward unless a separate
  precision smoke proves otherwise.
- Next required evidence is Stage1 checkpoint/validation, Stage2 completion,
  DAVIS metrics, four-column videos, and Stage1/Stage2 dpo_diag summaries.

## 2026-06-06 Reproducible HAL/H20/PAI Workflow Rule

From Exp8c onward, experiment code must not be patched only in a remote
terminal. HAL is the source of truth for code and PRD:

1. Modify experiment code, data-prep tools, launchers, PRD, and registry on HAL.
2. Commit and push to git.
3. Pull on H20 through Codex SSH.
4. Run PAI from the same pushed code, either by PAI `git pull --ff-only` or by
   rsyncing the H20-pulled code when PAI GitHub access is unstable.

Detailed rule: `PRD/15_reproducible_experiment_workflow.md`.

Current Exp8c PAI tracked entry points:

```text
tools/prepare_exp8c_gtwin_manifest.py
scripts/launch_exp8c_youtubevos_gtwin_d3comp_fullloss_s1s2_2000_davis_pai.sh
experiment_registry/exp08c_youtubevos_gtwin_d3comp_fullloss_davis_s1s2_2000/
```

## 2026-06-08 Exp9 / Exp10 / Exp11 Core Sequence

New target-domain experiments use numeric names only:

```text
Exp9  = log-ratio / normalized-gap DPO
Exp10 = region-local DPO
Exp11 = flow-prior consistency DPO
```

Do not use `Exp9a`, `Exp10a`, or A/B/C suffixes for this sequence.

Execution boundary:

- PAI is manual-launch only from a copy-paste command block.
- H20 should not start these trainings unless explicitly requested.
- Default launch is Exp9 only.
- Exp10 and Exp11 require explicit `RUN_EXPERIMENTS`.
- Exp11 is blocked until `reports/exp11_flow_prior_implementation_audit.md`
  passes.

Canonical folders:

```text
exp9_logratio_gap_dpo/
exp10_region_local_dpo/
exp11_flow_prior_consistency_dpo/
experiment_registry/exp09_logratio_gap_dpo/
experiment_registry/exp10_region_local_dpo/
experiment_registry/exp11_flow_prior_consistency_dpo/
```

Common hard rules:

- target-domain winner is GT / clean clip via `win_video_path`;
- loser is D3 generated loser via `final_loser_video_path`;
- reject PAI manifests containing `/home/nvme01`;
- use SFT-48000 DiffuEraser weights for YouTube-VOS/DAVIS;
- partial-mask inpainting uses ProPainter prior;
- DAVIS eval uses raw6, no PCM, no mask dilation, no Gaussian blur, hard comp;
- inpainting metrics use `tools/run_inpainting_metric_eval.py` and
  `inference/metrics.py`; VBench is not valid for this task.

## 2026-06-08 Exp11 GPU Availability Does Not Override Audit Block

PAI snapshot at 2026-06-08 14:19 CST:

```text
GPU0-3: only lightweight python3 processes, about 0.9 GiB each
GPU4-7: Exp9 `lingbot-worldmodel` training, about 66-67 GiB each on 4-6 and
        about 124 GiB on GPU7
```

Although GPU0-3 appear available, Exp11 must not be launched there yet. The
current launcher and PRD intentionally mark Exp11 as blocked because train-time
flow / ProPainter-prior / boundary consistency is not safely implemented in the
Stage1/Stage2 DPO loops. Running `RUN_EXPERIMENTS=exp11` should only write
`reports/exp11_flow_prior_implementation_audit.md` and stop unless a future
implementation audit passes. Do not bypass this with `EXP11_ENABLE_TRAINING=1`.

H20 Exp10 status at 2026-06-08 14:20 CST:

```text
host = H20
pid = 956576
log = /home/nvme01/H20_Video_inpainting_DPO_pai_sync_latest/logs/pipelines/exp10_region_local_dpo_s1s2_2000_davis_pai_gpus4_7_fp32_20260608_132413_h20_exp10_fp32.log
stage1_dir = /home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260608_132413_h20_exp10_fp32_exp10_region_local_dpo_s1_2000_davis_pai
status = Stage1 running, fp32 / no split-forward H20 profile
latest observed global_step = 210
dpo_diag rows = 22
stage2 = not started
```

The H20 Exp10 run is slow but healthy. It records `loss_region_mode=region`,
`gap_normalization=log_ratio`, raw and normalized win/lose gaps, clipped loser
gap, and region-local MSE fields in `dpo_diagnostics.csv`.

## 2026-06-08 PAI Direct SSH / Exp9 Live Status

HAL/Codex direct SSH to PAI is now verified through the `hj-pai-20260608`
ED25519 key recorded in `PRD/15_reproducible_experiment_workflow.md`.
Operational boundary: Codex may use direct SSH for PAI audits, log checks, and
file surveys. Starting or restarting PAI training still requires an explicit
user request.

PAI direct survey at 2026-06-08 14:55 CST:

```text
pai_host = dsw-753014-dc85766cb-4v2jj
pai_user = root
ssh = root@47.103.26.60 -p 22
workspace = /mnt/workspace/hj/nas_hj -> /mnt/nas/hj
active_sync_repo = /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp09_10_11_pai_sync
active_sync_commit_marker = 7d31184
```

Top-level PAI `/mnt/workspace/hj/nas_hj` code/data roots visible in the survey:

```text
H20_Video_inpainting_DPO
H20_Video_inpainting_DPO_exp09_10_11_pai_sync
H20_Video_inpainting_DPO_exp8c_pai_sync
conda_envs
data
external
official_repos
weights
world_model_phys
```

Current Exp9 PAI run:

```text
experiment = exp09_logratio_gap_dpo_s1s2_2000_davis_pai
run_version = 20260608_121243
launcher_pid = 3678169
process_name = lingbot-worldmodel
gpu = 4,5,6,7
stage = Stage1 running
stage1_dir = /mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260608_121243_exp9_logratio_gap_dpo_s1_2000_davis_pai
log = /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp09_10_11_pai_sync/logs/pipelines/exp09_logratio_gap_dpo_s1s2_2000_davis_pai_lingbot_worldmodel_gpus4_7_20260608_141428.log
latest_observed_global_step = 350
dpo_diag_rows = 56
latest_checkpoint = checkpoint-350 saved at 14:55 CST
```

The latest observed Exp9 diagnostics confirm the intended Exp9 setting:
`gap_normalization=log_ratio`, `lose_gap_clip_tau=1.0`,
`loss_region_mode=full`, raw and normalized gap fields present, and no
Traceback/OOM/SIGFPE/SIGTERM in the inspected log window. Earlier incomplete
checkpoints from the SIGTERM-interrupted run were moved under
`_bad_incomplete_checkpoints_*`; the current run resumed cleanly and now writes
complete checkpoints again.

## 2026-06-09 CST Exp9/10/11 Frame-Length Rule

Exp9 Stage1 finished on PAI with the old default `NFRAMES=16`, but the following
DAVIS validation failed before producing videos:

```text
ValueError: The effective video duration is too short. Please make sure that
the number of frames of video, mask, and priori is at least greater than 22
frames.
```

Root cause: `scripts/launch_exp09_10_11_pai.sh` still defaulted both training
and validation to 16 frames:

```text
NFRAMES=16
DAVIS_VIDEO_LENGTH=16
```

Follow-up PAI retry showed the existing D3 generated-loser training clips only
contain 16 frames:

```text
RuntimeError: Expected at least 24 frames under .../gt_win_cache/.../win,
found 16
```

Therefore the correct rule for the current non-regeneration Exp9/10/11 run is:

- existing D3 generated-loser training remains `NFRAMES=16`;
- DAVIS / ProPainter validation must use `DAVIS_VIDEO_LENGTH=24`;
- do not rerun DAVIS validation with `DAVIS_VIDEO_LENGTH=16`;
- do not fake 24-frame training by padding or repeating frames;
- training/validation frame-count parity can only be restored by regenerating
  D3 loser/winner/mask clips at a length greater than 22, which is explicitly out
  of scope for the current run.

The Exp9/10/11 launcher and experiment configs were corrected to default to
`NFRAMES=16` for existing D3 training and `DAVIS_VIDEO_LENGTH=24` for DAVIS
validation.

## 2026-06-09 CST PAI Non-Interactive Conda Prefix Rule

The first 24-frame PAI relaunch exited immediately at Stage1 with:

```text
conda not found; set CONDA_EXE or install Miniconda.
```

Root cause: the PAI diffueraser runtime is available as a full environment
prefix:

```text
/mnt/nas/hj/conda_envs/diffueraser/bin/python
```

but this non-interactive environment does not expose a `conda` executable. The
Stage1/Stage2 sbatch wrappers must accept `CONDA_ENV_PREFIX` as a runnable env
prefix by prepending `${CONDA_ENV_PREFIX}/bin` to `PATH` and setting
`CONDA_PREFIX`, instead of requiring `conda activate`.

Rule: for PAI nohup / setsid launches, always pass:

```text
CONDA_ENV_PREFIX=/mnt/nas/hj/conda_envs/diffueraser
```

and the sbatch wrapper should run from that prefix even when `conda` is absent.

## 2026-06-09 CST PAI SIGTERM Resume Fallback

During Exp10 Stage1 on PAI, the run reached step 1350 after complete
checkpoint-500 and checkpoint-1000 saves, then the distributed workers received
external `SIGTERM`:

```text
traceback : Signal 15 (SIGTERM) received by PID ...
```

This was not an OOM/SIGFPE/code exception. A full-state resume from
`checkpoint-1000` also received `SIGTERM` shortly after model weights loaded,
before optimizer/scheduler state loading finished. The PAI `nohang` log showed
high available host memory and no visible corrective action at the inspected
timestamps, so the exact external sender is not proven from local logs.

Fallback rule: when full accelerator-state resume repeatedly dies by external
`SIGTERM`, export the last complete checkpoint to DiffuEraser `last_weights`
format, launch a new continuation run with:

```text
POLICY_INIT_PATH=<exported_checkpoint_weights>
RESUME_FROM_CHECKPOINT=none
STAGE1_MAX_STEPS=<remaining_stage1_steps>
STAGE2_MAX_STEPS=2000
```

`POLICY_INIT_PATH` must initialize only the trainable policy UNet/BrushNet.
The frozen reference model must still come from the 48000-step SFT DiffuEraser
weights so DPO reward semantics are preserved. Do not set `REF_MODEL_PATH` to
the DPO checkpoint for this fallback.

`STAGE1_MAX_STEPS` is only for the replacement Stage1 continuation run. It must
not shorten Stage2; Stage2 remains a full 2000-step run unless a separate,
explicit Stage2 recovery plan says otherwise.

Implementation guard: `training/dpo/scripts/run_stage1.py` must not default to
`--resume_from_checkpoint latest` internally. Recovery correctness is checked
from the final printed command line: a policy-init fallback run must contain
`--policy_init_path <...>` and must not contain `--resume_from_checkpoint
latest`.

## 2026-06-09 CST PAI SIGTERM Process-Name Mitigation

Follow-up SIGTERM audit: Exp10 policy-init recovery was relaunched through a
foreground SSH session on GPUs 0-3, but rank0 still received `SIGTERM` at
step 40. This proves that the failure is not caused only by `nohup`, `setsid`,
or terminal disconnection.

The earlier `LINGBOT_PROCESS_NAME`, `PROCESS_TITLE`, and
`lingbot-worldmodel-stage*.py` entrypoint names are not sufficient by
themselves. The GPU workers are still spawned by the conda Python executable
unless the launcher explicitly runs accelerate through a renamed Python binary.
For PAI protected runs, create/use a copied Python executable such as:

```text
/mnt/nas/hj/conda_envs/diffueraser/bin/lingbot-worldmodel
```

and launch with:

```text
PYTHON_BIN=/mnt/nas/hj/conda_envs/diffueraser/bin/lingbot-worldmodel
DPO_ACCELERATE_PYTHON_BIN=/mnt/nas/hj/conda_envs/diffueraser/bin/lingbot-worldmodel
LINGBOT_PROCESS_NAME=lingbot-worldmodel
PROCESS_TITLE=lingbot-worldmodel
```

The stage sbatch wrappers must call `${PYTHON_BIN}` for
`training/dpo/scripts/run_stage*.py`, and the Python runner must invoke
`python -m accelerate.commands.launch` through
`DPO_ACCELERATE_PYTHON_BIN`. This is required so the actual torch distributed
GPU worker executable is no longer the generic `python` process name.

## 2026-06-09 CST PAI SIGTERM Still External After Process Rename

Follow-up audit after the process-name mitigation:

- Exp10 policy-init continuation was launched foreground over SSH on GPUs 0-3
  and still received external `SIGTERM` at about step 40.
- The actual worker Python executable was then changed to
  `/mnt/nas/hj/conda_envs/diffueraser/bin/lingbot-worldmodel`; the run still
  received external `SIGTERM` while saving `checkpoint-25`, leaving that
  checkpoint incomplete.
- The actual worker Python executable was then changed to
  `/mnt/nas/hj/conda_envs/diffueraser/bin/lingbotworld-phy`, matching the
  process name of long-lived existing PAI workers. The printed stage log
  confirmed:

```text
[dpo-stage1] python_runner=/mnt/nas/hj/conda_envs/diffueraser/bin/lingbotworld-phy
[dpo-stage1] accelerate_python=/mnt/nas/hj/conda_envs/diffueraser/bin/lingbotworld-phy
```

That run still received:

```text
traceback : Signal 15 (SIGTERM) received by PID 2672520
```

at about step 6. Local PAI evidence at inspection time did not show CUDA OOM,
host OOM, `SIGFPE`, or a Python code exception. `dmesg` had no OOM-kill record
for the inspected PIDs, and the visible `nohang` log did not prove it killed
the workers. Therefore the current blocker is an external PAI/DSW/admin-side
termination policy or sender that is not identifiable from the experiment code.

Do not keep relaunching Exp10/Exp11 on PAI until the external `SIGTERM` sender
is identified or the job/process is allowlisted. Repeated relaunches produce
partial runs and incomplete checkpoints. The administrator should inspect node
or platform logs around these CST timestamps and worker PIDs:

```text
2026-06-09 13:16:58 CST: Exp10 foreground SSH policy-init run, workers around 2665045-2665048.
2026-06-09 13:31:49 CST: Exp10 named Python run, workers around 2669008-2669011.
2026-06-09 13:38:50 CST: Exp10 lingbotworld-phy run, workers around 2672520-2672523.
```

Current verified experiment state:

- Exp9 completed Stage1, Stage2, Stage1 DAVIS validation, and Stage2 DAVIS
  validation under `RUN_VERSION=20260609_025331_d3n16_val24`.
- Exp10 original PAI Stage1 reached step 1350 with complete checkpoints at 500
  and 1000, then was externally terminated. The exported
  `checkpoint-1000_policy_init` loads successfully, but continuation attempts
  are also externally terminated.
- Exp11 remains implementation-blocked by the flow/prior consistency audit and
  must not be launched merely because GPUs are free.
## 2026-06-19 Exp20/21/22 Autoresearch Setup

Current best remains:

```text
Exp11 boundary outer b0.75 S2
```

New isolated branch/worktrees:

- HAL: `/home/hj/H20_Video_inpainting_DPO_exp20_autoresearch`
- PAI: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp20_autoresearch`
- branch: `research/exp20-adaptive-region-autoresearch-20260619`

New isolated tracks:

- Exp20: DiffuEraser-only scale-adaptive region-balanced DPO search.
- Exp21: multibackbone VideoDPO BR plumbing smoke matrix.
- Exp22: multimodel BR baseline asset/smoke preparation.

Status:

```text
PRECHECK_IMPLEMENTED_NOT_TRAINING_READY
```

No Exp20 training/sweep result has been claimed. Heavy PAI search is gated on
legacy parity, locked dev split, and recomputed SFT/Exp11 dev baselines.

## 2026-06-20 Exp20 First Fixed-Boundary Pilot

Exp20 has passed the safety gates through the first fixed-boundary PAI pilot:

```text
LEGACY_FULL_PARITY_PASSED
REAL_10STEP_SMOKE_PASSED
DEV_BASELINES_LOCKED
FIRST_WAVE_COMPLETED
```

The best dev-only fixed config is:

```text
P4: fixed_image_px radius=16, boundary_weight=2.0
PSNR=29.390553, SSIM=0.969074, LPIPS=0.018198, Ewarp=11.994790
```

This exceeds SFT, Exp11-S1, and Exp11-S2 on locked dev PSNR, but does not reach
`TARGET_DEV_PSNR=29.523336` and has mixed LPIPS/Ewarp. It is a candidate for
the next dev gate only; no adaptive search, region-balanced search, Stage2,
DAVIS50, or YouTubeVOS100 final evaluation has been started.

## 2026-06-20 Exp20 update

Exp20 fast search + equal-step completed negative: best equal-step PSNR candidate EQ_BF07 reached PSNR 29.393079 on locked dev, below TARGET_DEV_PSNR 29.523336 and with mixed LPIPS/VFID/TC tradeoffs. No long training, Stage2, DAVIS50, or YouTubeVOS100 final eval was launched.

## 2026-06-21 Exp20 multiseed shadow confirmation

Exp20 completed strict P0/P4/BF07 equal-step three-seed confirmation on locked search-dev and independent shadow-dev.

```text
COMPLETED_NEGATIVE_AFTER_MULTISEED_SHADOW
```

Summary:
- P4 remains a small search-dev signal but fails shadow-dev generalization.
- BF07 does not replace P4 and is worse on shadow-dev.
- AD04 remains a single-seed adaptive reference, not a promoted candidate.
- Codex visual review found no stable BF07 improvement over P4.

No 500-step gate, 1000/2000-step training, Stage2, DAVIS50, or YouTubeVOS100 final evaluation was launched. No further boundary radius/weight search is recommended.

## 2026-06-21 Exp23 GPU2/4/5/6 pair completion

Exp23 completed the first Phase A paired training run on PAI after avoiding
GPU7's persistent ghost allocation:

```text
pair_id = phaseA_scale1_pair001_outer2_gpus2456
fresh Exp11 twin = fresh_exp11_outer_b075
candidate = candidate_scale1_outer2_b075
status = STAGE1_STAGE2_PAIR_COMPLETED
gpus = 2,4,5,6
```

Both fresh Exp11 and candidate completed Stage1 2000 + Stage2 2000 and wrote
`last_weights`. No Exp23 `Phy` process remained after completion. DAVIS50
evaluation has not run yet, so this is a training-completion milestone, not a
quality result.

## Exp50 update - VOID_WEIGHTS_READY

- Time: 2026-06-30T14:04:06.075339+08:00
- Status: `VOID_WEIGHTS_READY`
- Evidence: `reports/exp50_void_weight_relay_ingest.md`
- Relay SHA match: yes, 52 / 52 files, missing 0, mismatch 0.
- Safety: no training, no inference, no GPU, no VOID positive claim.

## Exp50 env update - VOID_ENV_PARTIAL

- Time: 2026-06-30T14:20:58.202107+08:00
- Status: `VOID_ENV_PARTIAL`
- Evidence: `reports/exp50_void_env_smoke.md`
- Imports: 44 pass, 0 fail.
- CUDA smoke: no failures; small matmul/backward finite.
- Caveat: exact official pins are not matched; heavyweight CUDA packages were not reinstalled.
- Safety: no training, no inference, no full 5B model load, no VOID positive claim.

## Exp50 trainable audit update - VOID_TRAINING_FORWARD_HEAVY_BUT_POSSIBLE

- Time: 2026-06-30T14:24:40.037508+08:00
- Status: `VOID_TRAINING_FORWARD_HEAVY_BUT_POSSIBLE`
- Evidence: `reports/exp50_void_trainable_forward_audit.md`
- Finding: official trainable forward exists, but default training is heavy transformer fine-tuning, not out-of-box LoVI-DPO.
- Safety: no training, no inference, no official source modification, no VOID positive claim.

## Exp50 Gate8 update - VOID_VOR_QUADMASK_GATE8_READY

- Time: 2026-06-30T14:40:28.047764+08:00
- Status: `VOID_VOR_QUADMASK_GATE8_READY`
- Evidence: `reports/exp50_void_vor_quadmask_adapter.md` and `reports/exp50_void_vor_quadmask_visual_review.csv`
- Gate8: 8 rows, REAL/BLENDER 4/4, scene overlap False.
- VOR-Eval excluded: True
- Safety: no training, no inference, no hard comp, no VOID positive claim.
- Next gate: official inference smoke remains blocked until environment status is `VOID_ENV_READY`; current C status is `VOID_ENV_PARTIAL`.

<!-- EXP50_C2_ENV_REPAIR -->

### Exp50 C2 VOID Environment Repair - 2026-06-30T15:25:36+08:00

- Status: `VOID_ENV_PARTIAL`.
- Exact blockers: `VOID_ENV_BLOCKED_TORCH`, `VOID_ENV_BLOCKED_DEEPSPEED`.
- `VOID_ENV_READY` not reached; F0/F1/F2/G gates not run.
- Safety: no inference, no training, no optimizer step, no VOID official source modification.

## 2026-06-30 Exp50 VOID Env Relay Ingest

Status: `VOID_ENV_READY`.

HAL/H20 wheelhouse relay was ingested into the PAI Exp50 branch. The isolated PAI env `/home/hj/conda_envs/void_exp50_official_v2` now imports the required VOID/CogVideoX stack with torch `2.7.1+cu126` and CUDA runtime `12.6`. CUDA bf16 tiny smoke passed on GPU `None`. No training or inference was run. Next gate is F0 component load smoke.
