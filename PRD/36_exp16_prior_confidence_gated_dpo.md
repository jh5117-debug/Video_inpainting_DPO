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
stage1_500_limit100_completed
```

Still not done:

- Stage2 is not wired for Exp16 prior-confidence loss and must remain disabled.
- Full prior cache has not been built.
- Full 2000+2000 training has not been launched or authorized.
- DAVIS50 / YouTubeVOS100 Exp16 evaluation has not been run.
- This is not a final method result.

Artifacts:

```text
prior_cache = /mnt/nas/hj/H20_Video_inpainting_DPO/data/cache/exp16_propainter_prior_cache_limit100
prior_manifest = /mnt/nas/hj/H20_Video_inpainting_DPO/data/cache/exp16_propainter_prior_cache_limit100/manifests/exp16_train_with_prior_limit100.jsonl
stage1_500 = /mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260617_exp16_limit100_exp16_prior_confidence_s1_500_limit100_pai
```

Diagnostic summary:

```text
reports/exp16_dpo_diag_summary_limit100.md
```

## Decision Rule

Do not launch full Exp16 training until:

1. Stage2 receives the same real-prior latent-x0 wiring as Stage1.
2. A full prior cache plan is approved.
3. The Stage1 500 diagnostics are reviewed, especially high `implicit_acc` and
   high `loser_dominant_ratio`.
4. A small decode / validation check confirms that the Stage1 500 checkpoint is
   not visually harmful.

## Expected Evidence

If training passes:

- `exp16_prior_confidence_gated_dpo/dpo_diag/dpo_diagnostics.csv`
- DAVIS50 metric summary
- YouTubeVOS100 metric summary if wrapper is ready
- five-column visual comparison:
  GT / mask / SFT-48000 / Exp11 outer b0.75 S2 / Exp16
