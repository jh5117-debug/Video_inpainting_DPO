# Exp56-H20 Next Steps

Do not run 10-step from this lane.

Exp57 aggregator should read:

- `reports/exp56_h20_r5_onestep_summary.json`
- `reports/exp56_h20_r5_onestep_metrics.csv`
- `reports/exp56_h20_r5_onestep_visual_review.csv`
- `reports/exp56_h20_candidate_ranking.csv`

If continuing after Exp57, the next objective should preserve transition regions more directly. Object-only DPO with preservation was not enough; the blocker is still overlap / affected / boundary regression.
