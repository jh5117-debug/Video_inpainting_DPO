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

## Gate Result

Status:

```text
COMPLETED_NEGATIVE_STAGE1_GATES
```

All three Stage1-1000 gates completed on PAI:

- `exp17a_positive_s1_1000`
- `exp17b_saturation_s1_1000`
- `exp17c_combined_s1_1000`

DAVIS10 fixed-protocol summary:

| Method | PSNR | SSIM | strict mask PSNR | boundary PSNR | bbox PSNR | bbox SSIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exp11 boundary outer b0.75 S2 | 30.2950 | 0.9664 | 18.7651 | 24.7722 | 21.7774 | 0.7705 |
| SFT-48000 baseline | 29.6227 | 0.9616 | 18.0928 | 24.1247 | 21.1050 | 0.7390 |
| Exp17a positive S1-1000 | 29.7313 | 0.9632 | 18.2014 | 24.4509 | 21.2137 | 0.7466 |
| Exp17b saturation S1-1000 | 29.8542 | 0.9623 | 18.3243 | 24.4384 | 21.3366 | 0.7502 |
| Exp17c combined S1-1000 | 29.5117 | 0.9609 | 17.9818 | 24.4214 | 20.9941 | 0.7316 |

Decision:

```text
Best Exp17 variant = Exp17b saturation, but it does not beat Exp11.
No Stage1-2000 extension.
No Stage2.
Keep Exp17 as a negative ablation.
```

Diagnostic conclusion:

- `loser_dominant_ratio` remains high for all variants.
- Saturation gate did not meaningfully activate:
  `sat_weight_mean` stayed near 0.988 and `saturated_pair_ratio` stayed 0.0.
- Positive-side strengthening produced only isolated weak positives and did not
  stabilize the method.

Visual conclusion:

- Weak positives: Exp17a on `boat`; Exp17b on `breakdance` / `lucia`.
- Clear failures: `rhino`, `dog-agility`, `dance-jump`, `soccerball`, and
  Exp17c on `blackswan`.
- No Exp17 variant has enough qualitative evidence to replace Exp11.

Evidence:

```text
reports/exp17_davis10_gate_metric_summary.md
reports/exp17_davis10_gate_decision.json
reports/exp17_dpo_diag_summary.md
reports/exp17_visual_case_judgement.md
/home/hj/dpo-2-1-exp/exp17_saturation_positive_dpo_davis10_visuals/
```

## Required Evidence

- dpo_diagnostics.csv for every variant.
- DAVIS10 metric summary versus SFT and Exp11.
- Five-column visual comparison:
  GT / mask / SFT-48000 / Exp11 outer b0.75 S2 / Exp17.
- Codex visual judgement before any Stage1 2000 extension.

Evidence is complete for the Stage1-1000 gate. The extension condition failed.

## Guardrails

- Do not modify old Exp11 / Exp12 / Exp16 code.
- Do not modify shared `training/dpo`.
- Do not run Stage2 tonight.
- Do not use VBench.
