# DPO Diagnostic Summary

flags: COLLAPSE_RISK;DPO_SATURATED;LOSER_DOMINANT;WIN_GAP_EXPLODED
rows: 2082

| metric | mean | p50 | p90 | max |
| --- | --- | --- | --- | --- |
| dpo_loss | 0.038973 | 2.482e-12 | 0.00476 | 7.943036 |
| implicit_acc | 0.992795 | 1 | 1 | 1 |
| win_gap | 0.48734 | 0.468997 | 0.776293 | 1.320533 |
| lose_gap | 0.629038 | 0.622118 | 0.948834 | 1.381771 |
| raw_win_gap |  |  |  |  |
| raw_lose_gap |  |  |  |  |
| norm_win_gap |  |  |  |  |
| norm_lose_gap |  |  |  |  |
| norm_lose_gap_clipped |  |  |  |  |
| winner_abs_reg |  |  |  |  |
| winner_gap_reg |  |  |  |  |
| mse_w_over_ref_mse_w |  |  |  |  |
| mse_l_over_ref_mse_l |  |  |  |  |
| sigma_term | 0.98545 | 1 | 1 | 1 |
| kl_divergence | 0.279094 | 0.273492 | 0.429904 | 0.675576 |
| loser_dominant_ratio | 0.998894 | 1 | 1 | 1 |
| grad_norm | 3.164398 | 0.000518 | 8.843862 | 175.504849 |
| prior_loss |  |  |  |  |
| boundary_loss |  |  |  |  |
| flow_loss |  |  |  |  |

## Sources

- `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260526_172044_exp5_d2_comp_k4_stage1_full/dpo_diagnostics.csv`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260527_181634_exp5_d2_comp_k4_stage2_full/dpo_diagnostics.csv`
