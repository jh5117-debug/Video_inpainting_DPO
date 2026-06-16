# Exp15 OR Benchmark Status

Date: 2026-06-16

## Done

- DAVIS50 OR manifest created from DAVIS2017 foreground masks.
- YouTubeVOS100 OR manifest created from the fixed eval100 subset.
- Combined OR150 manifest created.
- DAVIS50 OR frames and masks verified on PAI/NAS.
- MiniMax-Remover repo cloned on HAL, official repo synced to PAI/NAS, and weights verified on PAI/NAS.
- MiniMax-Remover import fails in the shared DiffuEraser env because that env has `diffusers==0.29.2`; MiniMax needs a newer diffusers stack.
- COCOCO historical PAI/NAS weights and official repo path verified.
- Current DiffuEraser Exp11 outer b0.75 S2 BR metrics recorded as context.

## Next

- Run one-video smoke for ready methods.
- Build an isolated MiniMax env on local PAI storage or another non-shared location before MiniMax smoke.
- Launch full OR150 inference only after smoke passes.

## Blocked Methods

FloED, VACE, and VideoComp/VideoComposer do not yet have verified local PAI
repo+weights+OR wrappers. They must stay blocked until runnable.
