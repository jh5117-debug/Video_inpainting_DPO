# PRD 37: Exp17 Saturation-Aware DPO-Positive Region Loss

Date: 2026-06-17

## Current Decision

Paused:

- OR benchmark.
- BR / VideoPainter adapter.
- Adaptive normalization.
- Exp16 full prior cache / full training.
- Further Exp11 / Exp12 tuning.

Current best remains:

```text
Exp11 boundary outer b0.75 S2
```

Exp16 Stage1-500 validated implementation of prior-confidence gated latent-x0
loss, but it did not beat Exp11 on DAVIS10. Therefore Exp17 returns to the DPO
objective itself.

## Motivation

Exp11 outer b0.75 S2 improves DAVIS50 and YouTubeVOS100 over SFT-48000, but
dpo_diag still shows saturation and loser-dominant risk. Many DPO runs can have
low `dpo_loss` and high `implicit_acc` without clear visual improvement.

The suspected issue:

```text
pairwise DPO can keep increasing the preference margin by worsening the loser,
while the preferred/winner side is not protected strongly enough.
```

Exp17 tests whether positive-side preservation and saturation-aware pairwise
gating make the region-local DPO objective more stable.

## Base Loss

Exp17 inherits Exp11 outer b0.75 S2:

```text
g_w = log((m_w + eps) / (m_w_ref + eps))
g_l = log((m_l + eps) / (m_l_ref + eps))
g_l_clip = clip(g_l, max=1.0)

L_DPO = mean[-logsigmoid{-0.5 * 10 * (g_w - 0.25 * g_l_clip)}]
region = mask 1.0, outer boundary 0.75, outside 0.05
```

## Variants

### Exp17a: DPOP-style positive

```text
L_total = L_DPO + 0.05 * m_w + 2.0 * ReLU(g_w)
```

Goal: strengthen winner protection in mask / boundary regions.

### Exp17b: saturation-aware margin DPO

```text
margin_pref = 0.25 * g_l_clip - g_w
sat_weight = sigmoid(5.0 * (1.0 - margin_pref))
L_DPO_sat = mean[sat_weight.detach() * -logsigmoid{-0.5 * 10 * (g_w - 0.25 * g_l_clip)}]
L_total = L_DPO_sat + 0.05 * m_w + ReLU(g_w)
```

Goal: reduce pairwise gradient after the preference margin is already large.

### Exp17c: combined

```text
L_total = L_DPO_sat + 0.05 * m_w + 2.0 * ReLU(g_w)
```

Goal: combine positive preservation with saturation gating.

## First Gate

Run tonight:

```text
Exp17a Stage1 1000
Exp17b Stage1 1000
Exp17c Stage1 1000
DAVIS10 quick metric + five-column visual sanity for each
```

No Stage2 is launched automatically.

Extension rule:

```text
Only consider Stage1 2000 if a variant is not worse than Exp11 on DAVIS10
primary metrics, has more positive visual cases than failures, and has cleaner
dpo_diag.
```

## Required Evidence

- dpo_diagnostics.csv for every variant.
- DAVIS10 metric summary versus SFT and Exp11.
- Five-column visual comparison:
  GT / mask / SFT-48000 / Exp11 outer b0.75 S2 / Exp17.
- Codex visual judgement before any Stage1 2000 extension.

## Guardrails

- Do not modify old Exp11 / Exp12 / Exp16 code.
- Do not modify shared `training/dpo`.
- Do not run Stage2 tonight.
- Do not use VBench.
