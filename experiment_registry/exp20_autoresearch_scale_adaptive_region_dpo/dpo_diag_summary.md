# Exp20 DPO Diagnostics Summary

Completed stages:
- Legacy full parity: passed earlier.
- Real 10-step smoke: passed earlier.
- First wave, second fixed, region-balanced, adaptive, and equal-step trials all produced finite diagnostics except EQ_P4 first attempt, which OOMed once and then succeeded on retry.

Current diagnostic pattern:
- Loser-dominant ratio remains high for legacy/global candidates, usually near 1.0.
- Region-balanced RB candidates reduced loser_dominant_ratio to 0.0 in several runs, but this did not translate to higher PSNR.
- Max grad norms stayed finite in completed trials.

Key candidates:
- EQ_BF07: max_grad_norm 9.8334, loser_dominant_ratio 1.0.
- EQ_P4: max_grad_norm 10.0674, loser_dominant_ratio 1.0.
- EQ_AD04: max_grad_norm 10.5437, loser_dominant_ratio 1.0.
- EQ_RB08: max_grad_norm 9.5047, loser_dominant_ratio 0.0.

Multiseed shadow confirmation:
- P0 loser_dominant_ratio mean: 1.0.
- P4 loser_dominant_ratio mean: 1.0.
- BF07 loser_dominant_ratio mean: 1.0.
- No NaN/Inf was observed in completed P0/P4/BF07 equal-step multiseed runs.
- BF07 did not improve the loser-dominant pattern; the large boundary weight mainly keeps the same saturated loser-dominant behavior while hurting shadow-dev PSNR/LPIPS/VFID/TC.
