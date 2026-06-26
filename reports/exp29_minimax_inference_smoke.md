# Exp29 MiniMax Inference Smoke

Date: 2026-06-26

Status: `MINIMAX_INFERENCE_SMOKE_PASSED_WITH_VISUAL_QUALITY_RISKS`

MiniMax-Remover was run on four fixed DAVIS-style smoke samples using the
official local repository, pinned NAS weights, `num_inference_steps=12`, and
`iterations=6`.

Output root:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/minimax_inference_smoke_20260626`

## Protocol

- Repository: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/third_party/MiniMax-Remover/repo`
- Weights: `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/minimax_remover/current`
- Python: PAI system `python` with `diffusers` supporting `AutoencoderKLWan`
- GPU: right-side GPU0/GPU5 only
- Left CLI: read-only; no signals sent; GPU1-4 not used by Exp29
- Frames per sample: 9
- Resolution: 512 x 512
- Seed: 20260626

## Visual Summary

`davis_bear` is the only clearly useful OR smoke: the masked object is mostly
removed and the replacement background is plausible, though mildly smoothed.

`davis_bus` is technically valid but not a useful removal: the bus largely
remains and the output mostly repaints the masked vehicle region.

`davis_mallard-water` is a visual failure for OR utility: the duck remains and
the output introduces stable blue/black artifacts.

`davis_elephant` is technically valid but visually weak: the elephant remains
as a smoothed/washed-out structure and the output has visible white haze.

## Interpretation

This proves the local MiniMax inference stack can run, load weights, and
produce videos. It does not prove MiniMax is already a strong OR baseline.
MiniMax remains the strongest third-backbone true-adapter candidate because the
official architecture exposes a flow-style transformer forward, but OR
baseline utility is mixed and needs a larger task-matched review before being
used as a loser generator.

