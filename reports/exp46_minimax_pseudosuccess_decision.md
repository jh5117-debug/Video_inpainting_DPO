# Exp46 MiniMax Pseudo-Success Decision

Status: `MINIMAX_PSEUDOSUCCESS_STAGE2_NEGATIVE` / `MINIMAX_NOT_THIRD_BACKBONE_YET`

## Decision

Exp46 validates that the H20 pipeline can mirror Exp45 data, rewrite manifests, run BF16-safe DDP8 pseudo-success SFT, checkpoint, reload, and evaluate raw outputs. However, the 30-step pseudo-success SFT gate is negative. The run should stop here; `100-step` is not unlocked.

## Evidence

- Exp45 formal split was mirrored and validated at `64/24/24` with scene overlap `0`.
- BF16-safe P0-P7 preflight passed, including DDP8.
- Step0 baseline on search24/shadow24 was established using raw outputs.
- Pseudo-success SFT30 trained to `checkpoint-30` with finite loss/gradients.
- Step30 raw evaluation regressed both search and shadow.

| split | dFull PSNR | dMask PSNR | dBoundary PSNR | dOutside PSNR | dEwarp | visual worse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| search | -4.612642 | -0.548113 | -1.591353 | -4.812891 | -0.019463 | 24/24 |
| shadow | -3.366753 | -5.674479 | -3.636023 | -3.029058 | 0.021337 | 24/24 |

Codex opened and inspected the search and shadow Step30 contact sheets. Search shows subtle global tone/outside drift; shadow shows visible global brightness/color drift plus mask and boundary degradation.

## What Was Not Run

- GT-only SFT was not run.
- DPO was not run.
- 100-step pseudo-success SFT was not run.
- 300/500/1000/2000-step training was not run.
- VOR-Eval and hard comp were not used.

## Scientific Answer

The pseudo-success targets in their current form are not a safe SFT signal for MiniMax. The most likely blocker is target/output distribution mismatch plus insufficient preservation constraint: the model learns a global tone/style shift toward pseudo-success frames instead of a local object-removal adapter behavior. Shadow failure shows this is not only metric sensitivity; mask, boundary, outside, temporal metrics, and visual review all move in the wrong direction.

## Next Minimal Step

Return to data/objective design rather than longer training: build a stricter pseudo-success subset with near-identity outside regions, add explicit outside identity/consistency anchoring before any optimizer step, and run a 1/10-step overfit sanity check where outside delta must remain near zero before repeating 30-step.
