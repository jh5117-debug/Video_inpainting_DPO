# Exp26 Probe4 Moving Mask Visual Audit

Date: 2026-06-23

## Scope

Probe4 uses four VOR-Train-BG videos, each materialized as exactly 49 real
frames. I generated and opened contact sheets for all four samples:

- clean frames;
- mask-only frames;
- mask overlay;
- masked condition.

Visual evidence root:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/49f_probe_7f9ec40/probe4_mask_visuals`

Local copied contact sheets:

`reports/exp26_probe4_mask_visuals/`

## Result

`PROBE4_MASK_VISUAL_AUDIT_PASSED_WITH_SYNTHETIC_MASK_CAVEAT`

All four samples satisfy the technical mask gate:

- `first_frame_gt=true` is visible: frame 0 has no mask in every sample.
- Motion is smooth across the sampled frames.
- No obvious frame-order error, flicker, or centroid jump was visible.
- No mask disappears after frame 0.
- Area curves are smooth and bounded.

But the current mask generator should be described accurately:

- The masks are synthetic moving ellipses/circles.
- They are suitable as controlled moving BR stress masks.
- They are not semantic object-shaped occluders.
- One sample (`REAL_ENV266`) touches the frame boundary in 10 frames and should
  be tracked as an edge-touch caution case.

## Per-Sample Judgement

| sample | visual judgement |
| --- | --- |
| `REAL_ENV181_00006_007_01` | stable large ellipse, first-frame GT intact, no jump |
| `REAL_ENV266_00005_001_01` | stable small ellipse, first-frame GT intact, edge-touch caution |
| `REAL_ENV177_00004_003_02` | stable medium circle over moving person/park scene, no jump |
| `REAL_ENV222_00105_004_01` | stable medium ellipse with diagonal centroid motion, no jump |

## Gate Decision

The mask generator is locked as:

`exp26_videopainter_dpo_v2/configs/vp2_moving_mask_locked.json`

Gate16 is not started yet. The next required gate is official 49-frame
VideoPainter inference on Probe4 using the passed L0-L4 wrapper/checkpoint
path. If official inference produces systematic artifacts, we should fix the
VideoPainter wrapper before scaling sources; if it only exposes mask-shape
weakness, we should improve mask shape before Gate16.
