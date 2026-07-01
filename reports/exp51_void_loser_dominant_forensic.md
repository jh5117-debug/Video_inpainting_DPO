# Exp51 VOID Loser-Dominant Forensic

Time: 2026-07-01T10:33:01+08:00

Status: `VOID_LOSER_DOMINANT_CONFIRMED`

## Readback

Exp50 established VOID as an audited VOR-OR inference baseline and same-model loser generator candidate. It also validated the preference wrapper, zero-gap, and one-step path. The bounded 10-step vanilla LoVI-DPO gate was negative, but not because the model collapsed.

## Required Answers

1. Zero-gap sign/reference correctness: yes. Exp50 zero-gap had identical policy/reference state, near-zero winner/loser gaps, DPO loss near log(2), and frozen reference gradients.
2. 10-step margin growth: yes. Preference margin moved from 0.000000000 to 0.001065578; DPO loss moved from 0.693147182 to 0.693093896.
3. Margin source: final winner_gap=0.000072602, loser_gap=-0.000992976. Loser contribution accounts for 93.19% of final margin; winner contribution accounts for 6.81%.
4. Winner absolute loss improvement: weak/mixed. At final logged step, winner policy-reference loss delta is -0.000072602; positive means policy is worse than reference.
5. Loser degradation dominance: confirmed. At final logged step, loser policy-reference loss delta is 0.000992976, much larger in magnitude than winner movement.
6. Heldout metrics worse: mean SSIM -0.002341, mask PSNR -0.229878, boundary PSNR -0.063034; full PSNR was effectively flat at -0.000965.
7. Region most harmed: `mask_psnr` with delta -0.229878; object/mask suffered most, boundary also regressed.
8. Outside preservation: safe/mixed. Mean outside PSNR improved 0.043422 and outside L1 improved -0.010610; no systematic outside damage was visually observed.
9. Collapse: no. Exp50 summary marks no collapse, finite heldout outputs, and visual better/tie/worse 0/3/1.
10. Interpretation: current negative is recipe-specific, not a full VOID failure. The current objective learned by making loser predictions worse relative to reference more than by preserving/improving winner behavior.

## Core Numbers

| quantity | value |
|---|---:|
| loss_start | 0.693147182 |
| loss_end | 0.693093896 |
| margin_start | 0.000000000 |
| margin_end | 0.001065578 |
| final_winner_gap | 0.000072602 |
| final_loser_gap | -0.000992976 |
| loser_margin_share | 93.19% |
| winner_margin_share | 6.81% |

## Heldout Mean Deltas

| metric | delta |
|---|---:|
| full_psnr | -0.000965 |
| ssim | -0.002341 |
| mask_psnr | -0.229878 |
| affected_psnr | 0.019341 |
| boundary_psnr | -0.063034 |
| outside_psnr | 0.043422 |
| outside_l1 | -0.010610 |
| flicker | 0.026447 |

## Conclusion

Proceed with winner-preserving, loser-clipped, and local-only recipes. Do not scale vanilla LoVI-DPO from Exp50.
