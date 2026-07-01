# Exp57 Metric Summary

PAI one-step metrics have been produced for two Q2/T500/S0 adaptive cells.

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

PAI Exp57 one-step:

| Cell | Status | Full | Object | Overlap | Affected | Boundary | Outside | Visual |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ATS_SDPO_Q2_T500_S0 | NEGATIVE | 0.039160 | -0.337918 | -0.255698 | 0.108109 | -0.049336 | 0.075966 | 0/0/4 |
| ATS_LINEAR_Q2_T500_S0 | NEGATIVE | 0.035889 | -0.003580 | -0.395469 | 0.053643 | -0.085521 | 0.099602 | 0/0/4 |
