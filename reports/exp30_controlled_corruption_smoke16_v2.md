# Exp30 Controlled-Corruption Smoke16 V2

Status: `CONTROLLED_CORRUPTION_SMOKE16_LOW_YIELD`

- Candidate count: 16
- Technical valid count: 16
- Usable medium-hard/hard-plausible count: 5
- Classification counts: `{'TRIVIAL_BAD': 11, 'HARD_BUT_PLAUSIBLE': 2, 'MEDIUM_HARD_ELIGIBLE': 3}`
- Review CSV: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp30_vor_or_multimodel_minimax/smoke16_v2_20260627/reports/exp30_controlled_corruption_smoke16_v2_review.csv`
- Metrics CSV: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp30_vor_or_multimodel_minimax/smoke16_v2_20260627/reports/exp30_controlled_corruption_smoke16_v2_metrics.csv`

This is a controlled local corruption fallback, not a model-generated primary loser by itself. It can satisfy the fallback part of smoke16 but does not alone unlock Gate64 or MiniMax adapter training.
