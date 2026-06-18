# PRD 41: Exp19 Flow Strength and Warp Supervision

Date: 2026-06-18

## Motivation

Exp19b Stage2-500 proved that an isolated Stage2 flow-adapter wrapper can train
and that the adapter reads real flow. However, DAVIS10 showed an almost no-op
effect relative to Exp11 outer b0.75 S2. Before changing training, the inference
path must be calibrated because the disabled wrapper differed from the standard
Exp11 evaluator by about `0.009878` MAE in the first eval wrapper.

## Gates

1. Exp19-R0 inference parity:
   `adapter disabled ~= original Exp11 evaluator`.
2. Zero-training residual scale / confidence exponent sweep.
3. Real-flow causality versus zero, shuffled, and reversed flow.
4. Exp19c-light confidence-gated latent warp loss, only if R0 passes.
5. Exp19d motion-aware sampling, only if Exp19c passes the positive gate.

## Guardrail

If R0 parity does not reach the configured bf16 tolerance (`5e-4`), do not run
residual sweeps, causality, Exp19c training, Exp19d, DAVIS50, or any full-cache
training.

## 2026-06-18 PAI Result

Status:

```text
EXP19C_DAVIS10_COMPLETED_NEGATIVE_GATE
```

R0 fixed inference parity:

```text
disabled_vs_Exp11_MAE = 0.0
```

Best zero-training calibration:

```text
residual_scale = 0.5
confidence_exponent = 2.0
```

Real-flow causality passed only weakly. Real flow beat zero/shuffled/reversed
controls, but the effect was tiny:

```text
real-flow Ewarp delta vs disabled = -0.000124
```

Exp19c-light ran four 500-step continuations from the same Exp19b-500
checkpoint:

| Variant | lambda_warp | Status |
| --- | ---: | --- |
| lambda000 | 0.000 | complete |
| lambda005 | 0.005 | complete |
| lambda010 | 0.010 | complete |
| lambda020 | 0.020 | complete |

DAVIS10 summary:

| Method | PSNR | SSIM | LPIPS | Ewarp | strict mask PSNR | boundary PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT-48000 | 29.618218 | 0.961986 | 0.02203941 | 8.372417 | 18.320434 | 24.273454 |
| Exp11 outer b0.75 S2 | 29.829309 | 0.963257 | 0.02065550 | 8.330730 | 18.531525 | 24.657501 |
| Exp19b Stage2-500 | 29.829470 | 0.963257 | 0.02065455 | 8.330525 | 18.531685 | 24.657372 |
| lambda000 | 29.829031 | 0.963255 | 0.02065269 | 8.330644 | 18.531247 | 24.657456 |
| lambda005 | 29.829262 | 0.963255 | 0.02065183 | 8.330690 | 18.531478 | 24.657507 |
| lambda010 | 29.829166 | 0.963256 | 0.02065105 | 8.330801 | 18.531382 | 24.657439 |
| lambda020 | 29.829368 | 0.963257 | 0.02065228 | 8.330675 | 18.531584 | 24.657357 |

Decision:

```text
Do not start Exp19d.
Do not run DAVIS50.
Do not continue to 1000 / 2000 steps.
```

Reason: the best warp variant by Ewarp is lambda020, but it does not beat the
lambda=0 control on Ewarp and the absolute deltas are far below the positive
gate. Visual review found no clear better case over Exp11; all inspected
high/mid/low motion examples were ties. Exp19c is therefore a negative /
exploratory ablation, not a new mainline method.

Reports:

- `reports/exp19_inference_parity_repair.md`
- `reports/exp19_residual_scale_confidence_sweep.md`
- `reports/exp19r0_flow_causality_audit.md`
- `reports/exp19_motion_score_audit.md`
- `reports/exp19c_warp_loss_implementation_audit.md`
- `reports/exp19c_dpo_diag_summary.md`
- `reports/exp19c_davis10_metric_summary.md`
- `reports/exp19c_motion_bin_summary.md`
- `reports/exp19c_visual_case_judgement.md`
- `reports/exp19_refinement_final_report.md`
