# Exp15 MiniMax-Remover Paper Eval Protocol Audit

Source PDF: `/home/hj/Video Inpainting/Minimax-Remover.pdf`

## Findings

1. **DAVIS split**: the paper reports results on **all 90 DAVIS videos**, not the current Exp15 DAVIS50 subset. The text says: "all 90 DAVIS videos" and Table 2 is labeled DAVIS Dataset.
2. **Pexels split**: the paper evaluates on **200 randomly selected Pexels videos**. The text states these 200 Pexels videos are not contained in training data.
3. **Mask source**: training masks are produced with GroundedSAM2-style annotation. For Pexels evaluation, the paper explicitly says masks are extracted by GroundedSAM2. For DAVIS, the paper refers to DAVIS datasets and object-removal masks; it does not provide a frame-level manifest in the PDF text, so exact mask identity must be verified from code/release if available.
4. **PSNR / SSIM definition**: the paper says PSNR and SSIM evaluate **background preservation**. It does not spell out the exact implementation details in the extracted PDF text. The most compatible interpretation is background-region metrics outside the removal mask, not BR hard-comp full-frame metrics.
5. **Comp or raw**: the paper describes object-removal outputs and background preservation metrics; it does not state that outputs are hard-composited with GT before scoring. The compatible protocol is raw output, no comp.
6. **TC**: TC follows COCOCO and AVID, using CLIP-ViT-H/B-14 features to compute temporal consistency. Exp15's current `TC_bg` is a simple background temporal-difference proxy and is **not the paper TC**.
7. **VQ / Succ**: VQ and Succ are GPT-O3 evaluation results. VQ is a visual quality score; Succ is GPT-O3 judged removal success rate.
8. **Inference setting**: MiniMax reports 480p, frame length 81, and 6 sampling steps for its main fast setting. Baselines are evaluated with frame length 32; VideoComposer and FloED inputs are expanded, while other video inpainters use their default code-base frame lengths/resolutions.

## Current Exp15 Differences

| Aspect | MiniMax paper | Current Exp15 |
|---|---|---|
| DAVIS count | 90 videos | 50-video subset |
| Pexels | 200 videos | not run |
| Masks | DAVIS/GroundedSAM2-style object masks; exact released DAVIS manifest not confirmed | DAVIS2017 foreground annotations for selected 50 videos |
| PSNR/SSIM | background preservation; exact implementation not fully specified in PDF | strict `PSNR_bg`; `SSIM_bg_ignore_mask` proxy |
| TC | CLIP-ViT-H/B-14 feature TC following COCOCO/AVID | simple pixel temporal-difference `TC_bg` proxy |
| VQ/Succ | GPT-O3 | not run |
| Output metric frames | raw object-removal outputs implied | raw outputs, no comp |

## Conclusion

The current Exp15 DAVIS50 numbers are useful as an internal DAVIS50-subset OR benchmark, but they are **not directly comparable** to MiniMax Table 2. To align with the paper, we need DAVIS90, the paper-compatible TC implementation, and ideally the exact MiniMax release evaluation script or mask manifest.
