# PRD 40: Exp19 Boundary-Gated Flow-Adapter DPO

Date: 2026-06-18

## Motivation

Current best:

```text
Exp11 outer b0.75 S2
```

Exp11 improves mask-region quality and outer-boundary seams, but its DAVIS50 TC
is essentially unchanged from SFT-48000. Exp18 showed that forcing final outputs
to match warped propagated pixels is too rigid: non-oracle and oracle variants
both failed to beat Exp11. Exp19 therefore changes the role of flow from a hard
target into a light conditioning signal.

## Method

Exp19 adds zero-initialized residual flow adapters to DiffuEraser Stage2 while
keeping the Exp11 DPO objective unchanged.

Policy:

```text
Exp11 Stage2 base + flow adapter enabled
```

Reference:

```text
same Exp11 Stage2 base + flow adapter disabled
```

Base model parameters are frozen. Only the flow encoder, zero-conv residual
adapters, and alpha scales are trainable in the first gate.

## Flow Source

The intended flow source is ProPainter completed bidirectional flow from masked
inputs, without GT-error confidence and without using GT frames to estimate
confidence. Confidence is forward-backward consistency:

```text
C_flow = exp(-||F_f + Warp(F_b, F_f)||_1 / tau_flow) * valid_warp * source_valid
```

## Variants

| Variant | Gate | Extra loss | Purpose |
|---|---|---|---|
| Exp19a | `C_flow` | none | Test whether flow conditioning alone helps. |
| Exp19b | `C_flow * clip(mask + 0.75 * B_outer, 0, 1)` | none | Main boundary-gated candidate. |
| Exp19c | same as Exp19b | `0.05 L_warp` if safe | Diagnostic for latent temporal consistency. |

## Guardrails

- Do not modify Exp11 or shared training code.
- Do not train a second large flow-completion branch.
- Do not use GT to compute flow confidence.
- Do not use flow-warped pixels as final RGB targets.
- Do not run full 2000-step training until the limit=100 / 500-step gates pass.
- Do not use VBench.

## Current Status

```text
DAVIS10_EVAL_COMPLETED_NEGATIVE_GATE
```

The original architecture block has been recovered with an Exp19-only
hook-based wrapper:

```text
exp19_boundary_gated_flow_adapter_dpo/code/unet_motion_flow_adapter_wrapper.py
```

The wrapper injects directly at Stage2 motion-module outputs and does not use
the unsafe `additional_residuals` interfaces.

PAI gate result:

- limit100 ProPainter completed-flow cache: completed.
- zero-init / gradient preflight: passed.
- injected modules:
  - `mid_block.motion_modules.0`
  - `up_blocks.0.motion_modules.0`
  - `up_blocks.1.motion_modules.0`
- zero-init equality: passed (`mean_abs_diff = 0.0`).
- base model frozen: passed (`base_grad_norm = 0.0`).
- adapter gradient: non-zero.
- Exp19b Stage2 adapter-only 500 steps: completed.
- checkpoints: `checkpoint-250`, `checkpoint-500`, `last_weights`.

Exp19 inference wrapper:

```text
IMPLEMENTED_AND_VALIDATED
```

The standard DAVIS evaluator was not used to silently fall back to Exp11. An
Exp19-only inference wrapper now:

- loads Exp11 Stage1/Stage2 base weights;
- wraps the Stage2 UNet with the same three hook modules used in training;
- strict-loads `flow_adapter.pt`;
- constructs DAVIS completed-flow context from the input video and mask;
- queues per-window flow context through the DiffuEraser denoising loop;
- clears flow context after each forward.

Validation:

- strict load: passed;
- missing/unexpected adapter keys: none;
- adapter-disabled wrapper vs Exp11 MAE: `0.009878`;
- adapter-enabled vs disabled MAE: `0.009667`;
- real-flow vs shuffled-flow MAE: `0.009483`.

DAVIS10 completed:

| method | PSNR | SSIM | LPIPS | Ewarp | strict mask PSNR | boundary PSNR |
|---|---:|---:|---:|---:|---:|---:|
| SFT-48000 | 29.6181 | 0.9620 | 0.02204 | 8.3724 | 18.3203 | 24.2735 |
| Exp11 outer b0.75 S2 | 29.8295 | 0.9633 | 0.02065 | 8.3307 | 18.5317 | 24.6577 |
| Exp19b Stage2-500 | 29.8291 | 0.9633 | 0.02065 | 8.3306 | 18.5313 | 24.6574 |

TC was not computed because the TC backend attempted to download OpenCLIP from
Hugging Face and PAI network access failed. Ewarp was computed with local RAFT.

Decision:

```text
Do not expand Exp19b to 1000, DAVIS50, full cache, or 2000 steps.
```

Reason: Ewarp improves by only `0.000080` absolute relative to Exp11, far below
the 2% positive gate; PSNR, strict mask PSNR, and boundary PSNR are tiny
regressions; visual review found no reliable temporal improvement.

## 2026-06-18 Exploratory 2000 Follow-Up

At user request, Exp19b was nevertheless continued as an exploratory
longer-training check:

```text
start: Exp19b Stage2-500 flow_adapter.pt
continuation: +1500 adapter-only steps
total adapter steps: 2000
loss: Exp11 DPO loss only, lambda_warp = 0
```

Training completed on PAI and DAVIS50 evaluation completed. The evaluator row
still prints `Exp19b_stage2_500`, but the script loaded:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/exp19b_boundary_flow_adapter_s2_2000_exploratory_from500_limit100/last_weights/flow_adapter.pt
```

DAVIS50:

| Method | PSNR | SSIM | LPIPS | Ewarp | strict mask PSNR | boundary PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exp11 outer b0.75 S2 | 32.840213 | 0.971818 | 0.015339 | 7.181782 | 21.196763 | 26.441316 |
| Exp19b exploratory 2000 | 32.840122 | 0.971818 | 0.015340 | 7.181850 | 21.196671 | 26.441224 |

Decision:

```text
Do not continue Exp19b under this setup.
Current best remains Exp11 outer b0.75 S2.
```

The longer run did not validate the tiny DAVIS10 Ewarp trend. It is visually
safe but effectively no-op and slightly worse than Exp11 on the DAVIS50 metric
set.

## Exp19-R0 / Exp19c Refinement

Follow-up status:

```text
EXP19C_DAVIS10_COMPLETED_NEGATIVE_GATE
```

R0 fixed the inference-parity issue by matching the original Exp11 evaluator
protocol and resetting the global VAE latent-sampling RNG before paired
forwards. The calibrated disabled wrapper reached:

```text
disabled_vs_Exp11_MAE = 0.0
```

The best zero-training strength setting was:

```text
residual_scale = 0.5
confidence_exponent = 2.0
```

Real-flow causality passed only weakly: real flow beat zero/shuffled/reversed
controls, but Ewarp moved by only `0.000124` absolute on the R0 subset.

Exp19c-light added a confidence-gated latent warp consistency loss and ran four
500-step continuations from the same Exp19b-500 checkpoint:

| Variant | lambda_warp | DAVIS10 Ewarp | PSNR | LPIPS |
| --- | ---: | ---: | ---: | ---: |
| lambda000 | 0.000 | 8.330644 | 29.829031 | 0.02065269 |
| lambda005 | 0.005 | 8.330690 | 29.829262 | 0.02065183 |
| lambda010 | 0.010 | 8.330801 | 29.829166 | 0.02065105 |
| lambda020 | 0.020 | 8.330675 | 29.829368 | 0.02065228 |

Positive gate:

```text
FAIL
```

The best warp variant does not improve over the lambda000 continuation control
on Ewarp, and the absolute changes are far below the temporal gate. Visual
review of DAVIS10 contact sheets found no reliable better case over Exp11.
Exp19d, DAVIS50, and 1000/2000-step continuations were not launched.

Reports:

- `reports/exp19_isolated_wrapper_recovery_audit.md`
- `reports/exp19_isolated_wrapper_preflight.md`
- `reports/exp19b_dpo_adapter_diag_summary.md`
- `reports/exp19_inference_checkpoint_loading_audit.md`
- `reports/exp19_inference_preflight.md`
- `reports/exp19b_davis10_metric_summary.md`
- `reports/exp19b_visual_case_judgement.md`
- `reports/exp19_final_report.md`
