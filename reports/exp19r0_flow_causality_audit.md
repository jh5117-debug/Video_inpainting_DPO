# Exp19-R0 Flow Causality Audit

- best_residual_scale: `0.5`
- best_confidence_exponent: `2.0`
- causality_pass: `True`

| mode | PSNR | LPIPS | Ewarp | PSNR delta vs Exp11 | Ewarp delta vs Exp11 |
|---|---:|---:|---:|---:|---:|
| disabled | 32.441115 | 0.01314902 | 10.168823 | 0.000000 | 0.000000 |
| real | 32.441470 | 0.01314656 | 10.168699 | 0.000355 | -0.000124 |
| reversed | 32.441079 | 0.01314923 | 10.168817 | -0.000037 | -0.000006 |
| shuffled | 32.440994 | 0.01314800 | 10.168946 | -0.000121 | 0.000123 |
| zero | 32.441017 | 0.01314839 | 10.168903 | -0.000099 | 0.000080 |
