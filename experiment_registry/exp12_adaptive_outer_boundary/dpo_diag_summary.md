# DPO Diagnostics

Stage1 and Stage2 both produced `dpo_diagnostics.csv` with 201 rows.

Paths:

- Stage1: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260615_043419_exp12_adaptive_outer_exp12_batch_adaptive_outer_b075_s1s2_2000_s1_2000_davis_pai/dpo_diagnostics.csv`
- Stage2: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260615_043419_exp12_adaptive_outer_exp12_batch_adaptive_outer_b075_s1s2_2000_s2_2000_davis_pai/dpo_diagnostics.csv`

Final-interval pattern:

- `boundary_mode=outer`
- `implicit_acc=0.0` in the final sampled rows for both stages.
- `loser_dominant_ratio=0.0` in the final sampled rows.
- `adaptive_norm_win_gap` and `adaptive_norm_lose_gap` are near zero at the logged batch mean, as expected for batch z-score normalization.
- Stage2 final sampled row: `dpo_loss=2.2773`, `norm_win_gap=-0.0041`, `norm_lose_gap=0.1619`, `winner_gap_reg=0.3953`, `grad_norm=760.922`.

Interpretation:

The run is stable enough to complete, but the DPO preference signal is weak. This supports keeping Exp12 adaptive + outer b0.75 as an ablation rather than the current best method.
