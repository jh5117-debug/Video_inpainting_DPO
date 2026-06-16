# VideoPainter Adapter Gate2000 DPO Diagnostic Summary

Status: completed_training.

Rows: 201
First step: 1
Last step: 2000

Labels: DPO_SATURATED, LOSER_DOMINANT, GRAD_SPIKE_OBSERVED

| field | mean | last | min | max |
|---|---:|---:|---:|---:|
| loss | 0.0790636 | 0.0946371 | 0.0017734 | 0.705667 |
| dpo_loss | 0.0719959 | 0.085461 | 3.24712e-05 | 0.693147 |
| implicit_acc | 0.995025 | 1 | 0 | 1 |
| m_w | 0.141355 | 0.183523 | 0.0345296 | 0.886696 |
| m_l | 0.204957 | 0.222037 | 0.0538619 | 1.24889 |
| m_w_ref | 0.326197 | 0.326372 | 0.124829 | 1.20575 |
| m_l_ref | 0.337227 | 0.321274 | 0.13759 | 1.17744 |
| raw_win_gap | -0.184842 | -0.142849 | -0.603385 | 0 |
| raw_lose_gap | -0.13227 | -0.0992378 | -0.600007 | 0.522714 |
| norm_win_gap | -0.998701 | -0.575695 | -2.48605 | 0 |
| norm_lose_gap | -0.624133 | -0.369452 | -2.2248 | 0.924832 |
| norm_lose_gap_clipped | -0.624133 | -0.369452 | -2.2248 | 0.924832 |
| winner_abs_reg | 0.141355 | 0.183523 | 0.0345296 | 0.886696 |
| winner_gap_reg | 0 | 0 | 0 | 0 |
| mse_w_over_ref_mse_w | 0.426914 | 0.562311 | 0.0832369 | 0.999996 |
| mse_l_over_ref_mse_l | 0.639836 | 0.69111 | 0.108087 | 2.52144 |
| loser_dominant_ratio | 0.840796 | 1 | 0 | 1 |
| grad_norm | 2.95228 | 3.45657 | 0.0272639 | 80.3213 |
| mask_area_ratio | 0.190386 | 0.1675 | 0.151157 | 0.337639 |
| boundary_area_ratio | 0.0265506 | 0.02625 | 0.0216204 | 0.0340278 |
| outside_area_ratio | 0.783063 | 0.80625 | 0.634028 | 0.826019 |
| region_weight_sum | 5388.17 | 4914 | 4527.85 | 8436.75 |

Interpretation:

- The 2000-step gate completed and diagnostics remained finite.
- The mean DPO loss is very low and implicit accuracy is very high, so the run is marked `DPO_SATURATED`.
- `loser_dominant_ratio` is high, so the result should not be treated as a clean adapter win without DAVIS visual/metric evidence.
- A late-step grad-norm spike was observed, but training finished and saved `last_weights`.
