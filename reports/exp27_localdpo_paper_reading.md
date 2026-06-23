# Exp27 LocalDPO Paper Reading

Paper: Mind the Generative Details: Direct Localized Detail Preference Optimization for Video Diffusion Models  
arXiv: 2601.04068  
Official code: `https://github.com/1170300714/Local-DPO`

## Core Reading

LocalDPO is the closest prior to our current video-inpainting preference work. It constructs positive/negative video pairs from real clean videos and localized model-generated corruptions, then applies a region-aware DPO loss on the corrupted region. The paper already claims real-video winners, self-model local corruption losers, random spatiotemporal masks, localized preference learning, and hybrid global/local/SFT objectives.

Important method components:

- Positive sample: high-quality real video.
- Negative sample: same video with a localized region regenerated/corrupted by the base model.
- Mask: random spatiotemporal local mask, generated from connected/random Bezier-like regions in code.
- Latent fusion: generation is used inside the mask while outside-mask latent/content is re-injected from the original video path.
- Objective: standard global video DPO plus region-aware DPO plus SFT/winner term.
- Region loss: masked residuals normalized by mask occupancy and scaled by corruption strength `yita`/`alpha`.
- Training code has CogVideoX and Wan variants.

## Relevance To Our Project

LocalDPO overlaps strongly with generic claims such as "localized video DPO", "real winner plus generated loser", and "mask-aware DPO". Those claims are no longer safe for Exp27.

What remains distinct for Video_inpainting_DPO:

- Conditional video inpainting/object removal has an observed input video and a task mask.
- OR uses `condition = V_obj`, `winner = V_bg`, and foreground masks; this differs from random local corruption.
- BR uses mask core, outer seam/boundary, and observed outside context; LocalDPO does not explicitly optimize a hole/seam/context decomposition.
- LocalDPO does not model OR affected region, unaffected background preservation, object residual, copy-condition, or copy-winner failure modes.

## Required Baseline

Exp27 must implement a faithful LocalDPO-style inpainting baseline before claiming novelty. It should separate:

- task mask versus LocalDPO corruption mask;
- task-native OR/BR loser versus randomized local corruption loser;
- mask-only LocalDPO RA-DPO versus hole/boundary/context LoVI-style region decomposition.

## Parity Status

Official random-mask code was executed through Exp27 CPU parity and failed with:

`ValueError('cannot reshape array of size 1228800 into shape (480,640,3)')`

Source: `/home/hj/video_dpo_paper_code_cache/repos/Local-DPO/innerT2V/utils/random_mask_gen.py`

Status: `BLOCKED_OFFICIAL_CODE_RUNTIME_ERROR`. We will not fake a pass. Reproduction should either patch the official bug in an Exp27 adapter with a recorded patch or use a pinned environment where the official code runs unchanged.
