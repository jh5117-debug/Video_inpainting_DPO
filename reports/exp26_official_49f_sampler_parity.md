# Exp26 Official 49F Sampler Parity

Date: 2026-06-23

## Locked Formal Sampler

Current Exp26 formal source materialization uses:

- `num_frames = 49`;
- `stride = 1`;
- `offset = 0`;
- first 49 unique decoded source frames;
- no padding;
- no looping;
- no interpolation;
- no 13-frame fallback.

The strict implementation is:

`exp26_videopainter_dpo_v2/code/materialize_vp2_49f_sources.py`

The locked config is:

`exp26_videopainter_dpo_v2/configs/vp2_official_49f_sampler.json`

## Guard Status

The formal guard is already active:

- formal mode requires exactly 49 decoded frames;
- `--plumbing_only_13f` is the only allowed 13-frame path;
- `first_frame_gt` is a real toggle in dataset/mask plumbing;
- Probe4 materialization decoded `4/4` samples with exactly 49 unique frames.

## Official Native Caveat

The official VideoPainter training scripts use `--max_num_frames 49`; the
current Exp26 materializer therefore matches the formal frame-count contract.
This report does not yet claim full native inference parity for every internal
VideoPainter validation sampler. That requires the next Probe4 official
inference smoke to verify the same frame ordering through the wrapper.

## Decision

`FORMAL_49F_SAMPLER_LOCKED_PENDING_OFFICIAL_INFERENCE_SMOKE`

Gate16 must wait until Probe4 official inference produces valid 49-frame output
and its visual review is complete.
