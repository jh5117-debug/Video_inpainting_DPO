# Exp40 MiniMax Step0 Baseline
Status: `MINIMAX_STEP0_BASELINE_ESTABLISHED`. This milestone ran inference only; no training, no DPO, no hard comp, and no VOR-Eval usage. Raw MiniMax output is the primary baseline.
## Outputs
- PAI root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp40_minimax_psnr_safe_rescue/step0_baseline_20260629_0457`
- Local review pages: `reports/exp40_minimax_step0_review_pages/`
- Metrics: `reports/exp40_minimax_step0_baseline_metrics.csv`
- Visual review CSV: `reports/exp40_minimax_step0_visual_review.csv`

## Aggregate Metrics
| split | rows | full PSNR | mask PSNR | boundary PSNR | outside PSNR | outside MAE | temporal diff MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 64 | 23.965598 | 18.485359 | 19.395954 | 26.458319 | 10.077378 | 2.334655 |
| search | 24 | 25.043807 | 20.493872 | 21.409812 | 27.765446 | 8.877102 | 3.845744 |
| shadow | 24 | 26.209732 | 21.645338 | 24.277694 | 29.577002 | 7.695790 | 2.199505 |

## Codex Visual Review
Opened 14 midframe review pages and 28 temporal-strip pages covering all 112 Step0 baseline rows. Step0 shows a real baseline distribution: BLENDER mountain/product/river samples are comparatively stable, while many REAL indoor/person samples show object residuals, local color drift, ghosting, or dark/foggy fills. These are baseline weaknesses to improve against, not evidence of adapter success.

## Decision
Proceed to PSNR-safe SFT warmup only after preserving this Step0 baseline identity. Do not run DPO until SFT produces search/shadow improvements without boundary/outside damage.
