# Exp41 H20 MiniMax Official Protocol Audit

Status: `H20_MINIMAX_PROTOCOL_MATCHES_OFFICIAL`

Quality status: `PROTOCOL_ONLY_BASELINE_NOT_QUALITY_POSITIVE`

## Scope

- Branch: `research/exp41-h20-minimax-parallel-bf16-20260629`.
- Input HEAD: `fe10c86699ab62b26936ce206a221d158958b17f`.
- H20 run root: `/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp41_h20_minimax_parallel_bf16/protocol_audit_20260629_070149`.
- PAI policy: read-only only; no PAI GPU/signals/output mutation.
- Training launched in this milestone: `false`.
- Modified source code: `none`.

## Official Protocol Readback

H20 official MiniMax source and the local mirror agree on the executable protocol:

- README Quick Start imports `UniPCMultistepScheduler` and calls the pipeline with `num_inference_steps=12`, `iterations=6`, and `torch.float16` weights.
- `test_minimax_remover.py` uses the same scheduler, `num_inference_steps=12`, default `iterations=6`, and mask threshold `masks > 20` -> foreground mask.
- `pipeline_minimax_remover.py` has code defaults `num_inference_steps=50` and `iterations=16`, but those defaults are overridden by the official README/test executable examples.
- README feature text says "6 inference steps". Exp41 records this as a documentation ambiguity and ran a 6-step diagnostic probe. It is not adopted as the current executable official protocol.

## Current Exp40/H20 Protocol

The current Exp40 Step0 baseline runner matches the official executable README/test protocol for the fields that control inference:

- scheduler: `UniPCMultistepScheduler`.
- dtype: `float16`.
- steps/iterations: `12/6` for `official_readme_test`.
- seed: `20260629` in this smoke audit.
- raw output is primary: `true`.
- hidden/diagnostic comp used: `false`.
- winner/GT is used only as metric target and visual reference, not as the raw output.
- no ProPainter/EffectErase prior is used.
- mask polarity is correct: foreground object mask is 1, and the pipeline masks condition as `images * (1 - masks)`.

## Smoke Runs

| label | role | steps | iterations | train rows | search rows | runtime sec | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `official_readme_test` | official executable/current Exp40 match | 12 | 6 | 4 | 4 | 127.922 | `MINIMAX_STEP0_BASELINE_PENDING_CODEX_REVIEW` |
| `feature_6step_probe` | diagnostic README feature-text ambiguity | 6 | 6 | 4 | 4 | 117.081 | `MINIMAX_STEP0_BASELINE_PENDING_CODEX_REVIEW` |

## Aggregate Metrics

These are protocol-smoke metrics on 4 train + 4 search rows, not a promotion evaluation.

| label | split | full PSNR | mask PSNR | boundary PSNR | outside PSNR | outside MAE | temporal diff MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `official_readme_test` | `train4` | 28.014237 | 20.634074 | 21.408269 | 30.824739 | 4.555740 | 1.710896 |
| `official_readme_test` | `search4` | 27.342555 | 20.120630 | 21.166384 | 33.812574 | 3.230408 | 1.183885 |
| `feature_6step_probe` | `train4` | 28.027605 | 20.710534 | 21.539239 | 30.863828 | 4.466146 | 1.729045 |
| `feature_6step_probe` | `search4` | 27.547341 | 20.459429 | 20.962746 | 33.876004 | 3.137951 | 1.259875 |

## 6-Step Diagnostic Delta

The 6-step probe is not a replacement protocol. It was run only because README prose says "6 inference steps" while executable examples use 12.

| split | delta full PSNR | delta mask PSNR | delta boundary PSNR | delta outside PSNR | delta temporal diff MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train4` | 0.013368 | 0.076460 | 0.130969 | 0.039089 | 0.018149 |
| `search4` | 0.204787 | 0.338799 | -0.203638 | 0.063430 | 0.075990 |

## Visual Review

Codex opened the local contact sheets generated from all 16 midframe review sheets and all 16 temporal-strip sheets. It also decoded all 16 side-by-side mp4s with cv2: every video opened, first frame read succeeded, and every side-by-side mp4 had 17 frames at 2048x512, 8fps.

Visual counts across official 12-step and diagnostic 6-step rows:

- `PROTOCOL_VALID_BASELINE_OK`: 9
- `PROTOCOL_VALID_QUALITY_FAIL`: 3
- `PROTOCOL_VALID_QUALITY_TRADEOFF`: 4

Protocol observations:

- No mask-polarity reversal was observed.
- No GT/winner leakage into the raw output path was observed.
- No hidden comp path was used.
- Several rows still show baseline quality issues: terrain/shore hallucination, over-erasure/fog-like fill, and dark masked-region artifacts.
- Therefore this milestone passes protocol identity only. It does not establish MiniMax quality improvement or third-backbone evidence.

## Decision

`H20_MINIMAX_PROTOCOL_MATCHES_OFFICIAL` is accepted for the executable official README/test protocol.

Allowed next step: gated SFT-only bad-noise ladder may start after a fresh milestone readback and H20 GPU check.

Forbidden conclusions remain forbidden: no `UNIVERSAL_ADAPTER`, no `FINAL_SOTA`, no `TOP_CONFERENCE_NOVELTY_CONFIRMED`, and no MiniMax third-backbone claim from this audit.
