# Status

completed negative / no-op ablation

Exp19b was continued from the Stage2-500 adapter checkpoint for 1500
additional adapter-only steps, giving 2000 total adapter steps. DAVIS50 eval
completed on PAI.

Decision: do not continue this branch. The exploratory 2000 adapter is
effectively tied with Exp11 and slightly worse on PSNR, SSIM, LPIPS, strict
mask PSNR, boundary PSNR, and Ewarp.
