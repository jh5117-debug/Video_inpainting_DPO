# Metric Summary

DAVIS50 OR background-region metrics completed on PAI.

Protocol:

- no comp;
- raw method outputs;
- foreground mask is object/remove region;
- primary scores are background-region metrics (`mask == 0`);
- OR table is separate from BR hard-comp table.

| Method | Status | Success | PSNR_bg | SSIM_bg_ignore_mask | TC_bg |
|---|---|---:|---:|---:|---:|
| ProPainter | ok | 50/50 | 35.5274 | 0.9927 | 35.7664 |
| DiffuEraser SFT-48000 | ok | 50/50 | 28.6773 | 0.9686 | 28.8505 |
| Ours Exp11 outer b0.75 S2 | ok | 50/50 | 28.6795 | 0.9685 | 28.8682 |

Blocked methods are retained in the full summary with empty metrics:

- VideoComposer / VideoComp: no verified repo+weights+OR wrapper.
- CoCoCo: SD inpainting dependency incomplete.
- FloED: no verified repo+weights+OR wrapper.
- VideoPainter: no verified DAVIS2017 OR wrapper.
- VACE: no verified repo+weights+OR wrapper.
- MiniMax-Remover: isolated newer env required.

Full table:

```text
reports/exp15_or_davis50_quantitative_summary.csv
reports/exp15_or_davis50_quantitative_summary.md
```

Interpretation: ProPainter is the strongest runnable method for background
preservation. Exp11 outer b0.75 S2 is approximately tied with SFT-48000 under
this OR protocol and should not be claimed as the best OR method.
