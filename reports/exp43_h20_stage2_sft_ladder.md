# Exp43 H20 MiniMax Stage2 SFT Ladder

Status: `H20_EXP43_SFT_PARETO_MIXED`

This report intentionally blocks PASS/POSITIVE if visual review is pending.

| run | split | n | dPSNR | dMaskPSNR | dBoundaryPSNR | dOutsidePSNR | dLPIPS | dEwarp | visual reviewed | gate | blockers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| SFT-A_lr3em5_step30 | search | 24 | -5.833128230661999 | -4.674487775418862 | -4.700944147600658 | -7.594088453813615 | nan | 0.6460841968352801 | False | False | full_psnr -5.83313 < 0.08; mask_psnr -4.67449 < 0.05; boundary_psnr -4.70094 < -0.02; outside_psnr -7.59409 < -0.02; LPIPS delta nan > 0.001; Ewarp delta 0.646084 > 0.05; visual review pending; no metric-only promotion |
| SFT-A_lr3em5_step30 | shadow | 24 | -6.55060498000691 | -4.223185495799285 | -5.373455771430662 | -8.45318893655187 | nan | 0.5934015673112469 | False | False | full_psnr -6.5506 < 0.08; mask_psnr -4.22319 < 0.05; boundary_psnr -5.37346 < -0.02; outside_psnr -8.45319 < -0.02; LPIPS delta nan > 0.001; Ewarp delta 0.593402 > 0.05; visual review pending; no metric-only promotion |

Claim boundary:

- Technical training/evaluation completion is not scientific positive.
- MiniMax third-adapter evidence remains blocked until shadow metrics and visual review pass.
- No universal adapter, final SOTA, or top-conference novelty claim is made here.
