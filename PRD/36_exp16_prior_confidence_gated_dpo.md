# PRD 36: Exp16 Prior-Confidence Gated DPO

Date: 2026-06-17

## Current Scope

This PRD pauses the side branches and starts one new mainline attempt:

- OR is paused.
- BR / VideoPainter adapter work is paused.
- Adaptive normalization variants are paused.
- Exp11 / Exp12 tuning is paused.

Current best remains:

```text
Exp11 boundary outer b0.75 S2
```

Exp16 starts from that setting and adds a real ProPainter-prior confidence gate.

## Motivation

Exp11 outer b0.75 S2 already shows that region-local and boundary-aware DPO is
better than full-frame DPO. However, DiffuEraser uses ProPainter prior during
inference while the current DPO objective does not know where that prior is
reliable.

This can create two bad behaviors:

- in reliable ProPainter-prior regions, DPO may overwrite good propagated
  content;
- in unreliable prior regions, the model may trust poor propagation and produce
  fog, paste-like patches, grid artifacts, or boundary discontinuities.

Exp16 therefore tries to make DPO prior-aware rather than merely mask-aware.

## Method

Name:

```text
Prior-Confidence Gated DPO
```

Core idea:

- reliable prior region: keep predicted clean latent close to ProPainter prior;
- unreliable prior region: let diffusion / GT preference generate;
- outer boundary: keep seam consistency.

Confidence definition for the first version:

```text
P = ProPainter prior
x_GT = GT clean video
M = mask, 1 = hole
err_prior = mean_abs(P - x_GT) over RGB
C_prior = exp(-alpha * normalize(err_prior))
alpha = 5.0
```

Regions:

```text
M_reliable = M * C_prior
M_generate = M * (1 - C_prior)
boundary_outer = dilate(M) - M
```

Base loss inherited from Exp11 outer b0.75 S2:

```text
g_w = log((m_w + eps) / (m_w_ref + eps))
g_l = log((m_l + eps) / (m_l_ref + eps))
g_l_clip = clip(g_l, max=1.0)

L_DPO = mean[-logsigmoid{-0.5 * 10 * (g_w - 0.25 * g_l_clip)}]
L_base = L_DPO + 0.05 * m_w + ReLU(g_w)
```

Exp16 extra loss:

```text
L_prior = |z_hat_x0 - z_prior| over M_reliable
L_gen = |z_hat_x0 - z_gt| over M_generate
L_boundary_extra = |z_hat_x0 - z_gt| over boundary_outer

L_total =
    L_base
  + 0.1 * L_prior
  + 0.05 * L_gen
  + 0.1 * L_boundary_extra
```

Important guardrail:

```text
Exp16 must use real ProPainter prior frames and predicted clean latent x0.
Frozen-reference epsilon proxy is not allowed.
```

## Fixed Setting

Training:

- win = GT clean video
- lose = generated loser
- mask = partial mask from manifest
- base weights = SFT-48000 DiffuEraser
- prior = real ProPainter prior cache

Evaluation:

```text
DAVIS50 / YouTubeVOS100
raw6
D+G off
no PCM
no mask dilation
no Gaussian blur
hard comp
frame-wise metric
metric backend = inference/metrics.py
VBench = false
```

## Current Implementation Status

Implemented in isolated folder:

```text
exp16_prior_confidence_gated_dpo/
```

Completed in the current small gate:

- context audit;
- prior-cache builder;
- manifest loader that requires real prior paths;
- confidence-map helper;
- x0 reconstruction helper for epsilon / v-prediction / sample schedulers;
- latent prior / generation / boundary extra loss helper;
- preflight script;
- PAI launcher that blocks if prior manifest is missing;
- limit=100 real ProPainter prior cache on PAI;
- confidence-map audit on the limit=100 cache;
- Stage1 preflight with real prior frames, VAE latent targets, reconstructed
  predicted latent x0, and one dpo_diag row;
- Stage1 500 training on the limit=100 cache.

Current status:

```text
stage1_500_limit100_davis10_visual_sanity_completed
```

Still not done:

- Stage2 is not wired for Exp16 prior-confidence loss and must remain disabled.
- Full prior cache has not been built.
- Full 2000+2000 training has not been launched or authorized.
- DAVIS50 / YouTubeVOS100 full Exp16 evaluation has not been run.
- This is not a final method result.

Artifacts:

```text
prior_cache = /mnt/nas/hj/H20_Video_inpainting_DPO/data/cache/exp16_propainter_prior_cache_limit100
prior_manifest = /mnt/nas/hj/H20_Video_inpainting_DPO/data/cache/exp16_propainter_prior_cache_limit100/manifests/exp16_train_with_prior_limit100.jsonl
stage1_500 = /mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260617_exp16_limit100_exp16_prior_confidence_s1_500_limit100_pai
stage1_500_eval_hybrid = /mnt/nas/hj/H20_Video_inpainting_DPO/experiments/hybrid/20260617_exp16_stage1_500_limit100_dpoS1_sftS2/last_weights
```

Diagnostic summary:

```text
reports/exp16_dpo_diag_summary_limit100.md
reports/exp16_stage1_500_dpo_diag_summary.md
```

## 2026-06-17 DAVIS10 Visual Sanity

Small-scale DAVIS10 sanity has been completed. This is not a final Exp16 result;
it checks whether the Stage1-500 limit100 checkpoint has any positive signal
before committing to full prior cache or full training.

Output:

```text
PAI: /mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp16_stage1_500_visual_sanity_davis10
HAL: /home/hj/dpo-2-1-exp/exp16_stage1_500_visual_sanity_davis10
```

Exp16 was evaluated as:

```text
DPO-S1 Stage1-500 + SFT-48000 Stage2 hybrid
```

The Stage1-only checkpoint cannot be loaded directly by the DAVIS evaluator
because it lacks the Stage2 motion config.

Protocol:

```text
raw6, no PCM, no mask dilation, no Gaussian blur, hard comp, frame-wise metric, no VBench
```

Metric sanity:

| Method | PSNR | SSIM | strict mask PSNR | boundary PSNR | bbox PSNR | bbox SSIM |
|---|---:|---:|---:|---:|---:|---:|
| SFT-48000 | 29.8193 | 0.9625 | 18.2894 | 24.2926 | 21.3016 | 0.7380 |
| Exp11 outer b0.75 S2 | 30.1736 | 0.9644 | 18.6437 | 24.5907 | 21.6559 | 0.7513 |
| Exp16 Stage1-500 | 29.9460 | 0.9642 | 18.4161 | 24.5280 | 21.4284 | 0.7562 |

Visual judgement:

- Positive / weak signal: `lucia`, `dance-jump`, `soccerball`.
- Roughly tied: `bear`, `kite-surf`.
- Worse than Exp11: `boat`, `rhino`, `dog-agility`, `blackswan`, `breakdance`.

Interpretation:

```text
Exp16 Stage1-500 has weak positive signal and validates implementation, but it
does not beat Exp11 outer b0.75 S2.
```

Reports:

```text
reports/exp16_stage1_500_davis10_metric_summary.md
reports/exp16_stage1_500_visual_case_judgement.md
reports/exp16_confidence_diagnostic_fix_report.md
reports/exp16_confidence_limit100_offline_summary.md
```

## Confidence Diagnostic Fix

The original `reliable_area_ratio` / `generate_area_ratio` fields were
area-style nonzero statistics and were not sufficient to interpret the
confidence split. New fields have been added for future diagnostics:

```text
prior_conf_mean_inside_mask
prior_conf_std_inside_mask
prior_conf_p10_inside_mask
prior_conf_p50_inside_mask
prior_conf_p90_inside_mask
reliable_weight_mass
generate_weight_mass
reliable_generate_mass_sum
confidence_alpha
```

Offline recomputation on the limit100 cache:

| Field | Mean |
|---|---:|
| prior_conf_mean_inside_mask | 0.656014 |
| prior_conf_std_inside_mask | 0.264408 |
| prior_conf_p10_inside_mask | 0.239536 |
| prior_conf_p50_inside_mask | 0.725268 |
| prior_conf_p90_inside_mask | 0.940553 |
| reliable_weight_mass | 0.656014 |
| generate_weight_mass | 0.343986 |
| reliable_generate_mass_sum | 1.000000 |

The corrected confidence diagnostics are healthy, but the Stage1-500 visual
result is not strong enough to justify full training immediately.

## Decision Rule

Do not launch full Exp16 training until:

1. Stage2 receives the same real-prior latent-x0 wiring as Stage1.
2. A full prior cache plan is approved.
3. The Stage1 500 diagnostics are reviewed, especially high `implicit_acc` and
   high `loser_dominant_ratio`.
4. A small decode / validation check confirms that the Stage1 500 checkpoint is
   not visually harmful.

Current decision after DAVIS10 sanity:

```text
Do not launch full prior cache or Stage1 2000 yet.
If continuing Exp16, first reduce or schedule lambda_prior/lambda_gen and rerun
a small gate after user confirmation.
```

## Expected Evidence

If training passes:

- `exp16_prior_confidence_gated_dpo/dpo_diag/dpo_diagnostics.csv`
- DAVIS50 metric summary
- YouTubeVOS100 metric summary if wrapper is ready
- five-column visual comparison:
  GT / mask / SFT-48000 / Exp11 outer b0.75 S2 / Exp16
