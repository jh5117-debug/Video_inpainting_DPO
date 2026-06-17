# PRD 34: OR Eval Protocol and MiniMax Alignment

Date: 2026-06-17

## MiniMax Paper Protocol

MiniMax-Remover Table 2 reports:

- DAVIS: all 90 DAVIS videos;
- Pexels: 200 randomly selected Pexels videos;
- inference: 480p, frame length 81, 6 sampling steps for MiniMax main setting;
- baseline evaluation frame length: 32, with VideoComposer/FloED expanded when needed;
- PSNR/SSIM: background preservation metrics;
- TC: follows COCOCO/AVID with CLIP-ViT-H/B-14 features;
- VQ/Succ: GPT-O3 evaluation.

## Exp15 Current Protocol

Exp15 fixed DAVIS50 is an internal subset benchmark:

- DAVIS50 only, not DAVIS90;
- raw method outputs, no comp;
- foreground mask nonzero means remove/object region;
- `PSNR_bg`: strict background pixels (`mask == 0`);
- `SSIM_bg`: implemented as `SSIM_bg_ignore_mask`, foreground zeroed in both images;
- `TC_bg_pixel_proxy`: pixel temporal-difference proxy, not MiniMax paper TC;
- LPIPS/VFID: not available for current OR protocol;
- VQ/Succ: not run.

## Can We Compare To MiniMax Table 2?

Not directly.

Reasons:

1. MiniMax uses DAVIS90; Exp15 uses DAVIS50.
2. MiniMax TC is CLIP-feature based; Exp15 TC is a pixel proxy.
3. MiniMax includes GPT-O3 VQ/Succ; Exp15 does not.
4. The paper PDF does not fully specify the exact SSIM implementation and DAVIS mask manifest.

## Required For Paper-Compatible Alignment

- Use DAVIS90 manifest: `exp15_or_benchmark_davis90/manifests/davis90_or_manifest.csv`.
- Implement or import COCOCO/AVID CLIP-feature TC.
- Add GPT-O3 VQ/Succ evaluation or mark unavailable.
- Verify exact released MiniMax masks / evaluation script if available.
