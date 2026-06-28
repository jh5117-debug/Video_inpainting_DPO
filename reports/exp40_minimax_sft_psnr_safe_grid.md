# Exp40 MiniMax PSNR-Safe SFT Warmup Grid

Status: `MINIMAX_SFT_PSNRSAFE_NEGATIVE`

Run root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp40_minimax_psnr_safe_rescue/sft_psnr_safe_grid_20260629_0524`

Raw output remained the primary evaluation output. Diagnostic comp was not used.

Result: no recipe qualifies for 100-step continuation or DPO-after-SFT. All recipe aggregate deltas are negative on search.

| recipe | full dB | mask dB | boundary dB | outside dB | temporal proxy | full-positive rows | safe-positive rows | large-artifact rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `SFTmC_S0_lr3em05` | -1.816781 | -1.634597 | -1.899575 | -2.624405 | 0.525111 | 5 | 1 | 2 |
| `SFTmA_S0_lr3em05` | -2.303278 | -1.711663 | -1.975436 | -3.365938 | 0.584767 | 5 | 1 | 3 |
| `SFTmB_S0_lr3em05` | -2.306074 | -1.666656 | -1.952630 | -3.355257 | 0.595906 | 5 | 1 | 3 |
| `SFTmD_S0_lr3em05` | -2.395389 | -2.186769 | -2.440809 | -3.140433 | 0.414593 | 5 | 0 | 4 |
| `SFTmC_S0_lr0.0001` | -6.736656 | -5.878329 | -5.605923 | -8.168581 | 0.938385 | 0 | 0 | 16 |
| `SFTmD_S0_lr0.0001` | -6.801210 | -6.469689 | -6.300287 | -7.698793 | 1.156952 | 1 | 0 | 14 |
| `SFTmB_S0_lr0.0001` | -6.963161 | -5.897037 | -5.718433 | -8.264182 | 0.436655 | 0 | 0 | 15 |
| `SFTmA_S0_lr0.0001` | -7.172223 | -6.615492 | -6.301120 | -8.013239 | 0.475373 | 0 | 0 | 17 |
| `SFTmB_S0_lr0.0003` | -10.849289 | -9.839992 | -9.675967 | -12.674501 | 7.373321 | 0 | 0 | 24 |
| `SFTmA_S0_lr0.0003` | -13.827806 | -10.202384 | -10.424003 | -16.048283 | 7.137381 | 0 | 0 | 24 |
| `SFTmC_S0_lr0.0003` | -14.573955 | -10.318876 | -11.124060 | -17.201737 | 28.548663 | 0 | 0 | 24 |
| `SFTmD_S0_lr0.0003` | -15.079110 | -10.781672 | -11.678147 | -17.713054 | 24.738019 | 0 | 0 | 24 |

Visual review:
- Opened representative best-case and worst-case temporal strips copied under `reports/exp40_minimax_sft_psnr_safe_grid_review_assets/`.
- Best-looking PRODUCT004 rows show local waterline/bubble changes but not a recipe-level quality win.
- Worst high-LR mountain row shows obvious noisy/color collapse across frames.
- Since every aggregate recipe fails, no PASS/POSITIVE/DATA_READY style promotion is made.

Decision:
- `MINIMAX_SFT_PSNRSAFE_NEGATIVE`.
- Do not run 100-step.
- Do not run DPO-after-SFT.
- MiniMax remains plumbing-positive but recipe not quality-positive.
