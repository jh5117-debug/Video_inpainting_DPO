# Exp57 Metric Summary

H20 one-step metrics have been produced for four Q2/T500/S0 adaptive cells.

Zero-gap sanity:

- status: `EXP57_ADAPTIVE_ZERO_GAP_PASS`
- train preference loss: 0.693147
- heldout preference loss: 0.693147
- train / heldout preference margin: 0.0
- safe lambda global: 0.0469255468
- peak VRAM allocated: 20.016838 GiB

Source metrics read back:

- Exp56-H20 `R5_Q2_T500_S0`: full PSNR +0.013859, object PSNR +0.956095, overlap PSNR -0.153271, affected PSNR -0.084209, boundary PSNR -0.047360, outside PSNR +0.047483.
- Exp56-H20 `R5_HALF_Q2_T500_S0`: full PSNR +0.012121, object PSNR +0.667133, overlap PSNR -0.139584, affected PSNR -0.113662, boundary PSNR -0.069402, outside PSNR +0.047722.

H20 Exp57 one-step:

| Cell | Status | Full | Object | Overlap | Affected | Boundary | Outside | Visual |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ATS0_Q2_T500_S0 | NEGATIVE | -0.113495 | -0.598885 | -0.593289 | -0.213889 | -0.124392 | -0.007378 | 1/0/3 |
| ATS_STRICT_Q2_T500_S0 | NEGATIVE | -0.094041 | -0.473428 | -0.581142 | -0.146322 | -0.081617 | -0.005963 | 1/0/3 |
| ATS_HALFLR_Q2_T500_S0 | NEGATIVE | -0.104624 | -0.773073 | -0.707366 | -0.160285 | -0.150837 | -0.002763 | 0/0/4 |
| ATS_NODPO_Q2_T500_S0 | NEGATIVE | -0.094423 | -0.786377 | -0.563998 | -0.233663 | -0.123053 | -0.001418 | 0/0/4 |
