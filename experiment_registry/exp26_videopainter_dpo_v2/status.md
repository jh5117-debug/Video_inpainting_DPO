# Exp26 VideoPainter DPO v2 Status

- branch: `research/exp26-videopainter-dpo-v2`
- worktree: `/home/hj/H20_Video_inpainting_DPO_exp26_videopainter`
- status: `VP2_STATIC_FIXES_STARTED`
- copied source: `exp14_adapter_videopainter/`
- current scope: static v2 trainer fixes and unit tests
- GPU work: not started
- DAVIS50: not started

Initial fixes:

- official optimizer fields exposed;
- `noised_image_dropout` wired into image latent preparation;
- first-frame consistency helper added;
- native 49-frame policy enforced, 13F requires plumbing flag;
- formal mode now rejects 16-frame inputs instead of trimming them to 13;
- `--first_frame_gt` / `--no-first_frame_gt` now controls first-frame
  consistency;
- official optimizer/scheduler parser added for locked parity config;
- `itertools.cycle(loader)` removed;
- loser-dominant definition aligned with project diagnostics;
- strict checkpoint reload helper added.
# Exp26 Registry Status

L0_L4_PASSED
FORMAL_49F_SOURCE_BLOCKED

## 49F Source Diagnostic

- Source root: `/mnt/workspace/hj/nas_hj/data/external/ytbv_2019_full_resolution/train`
- Required valid candidates: 640
- Valid candidates: 0
- Failed candidates: 3471
- Max frame count seen: 36
- Max mask count seen: 36
- Gate64 official baseline self-loser generation: not launched.

Reason: active YouTube-VOS source is a sparse extraction and does not satisfy formal 49-frame input requirements.

## VOR-BG Source-Only Fallback

- status: `VOR_BG_SOURCE_SPLIT_LOCKED_PENDING_EXTRACTION_MASKS`
- source: VOR-Train BG clean videos
- train/search/shadow: 128 / 32 / 32
- split isolation: scene-group disjoint
- overlaps: train/search 0, train/shadow 0, search/shadow 0
- manifest hashes:
  - train `68a862c18ad46271757d96c6b5483c76aeb2cfdfb5cae21a452e215b735daaa6`
  - search `41c6571d26e4a5130818dd50fbbe1314c9d953284561a3cd20f630572f7c2a71`
  - shadow `0338dba1513cfe0e5dd85cbf793b3782902b981ac9610b0e472c6a048f738c02`

Gate64 has not started. Selected sources still require selective extraction,
exact 49-frame decode audit, and generated moving BR masks.
## 2026-06-23 Overnight Autonomous Controller

- Status: `OVERNIGHT_WAITING_GPU_OFFICIAL_PROBE_PENDING`.
- Runtime controller on PAI:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623`.
- CPU mask distribution audit completed and recorded:
  `reports/exp26_br_mask_distribution_audit_fast512.md`.
- Probe4 masks remain plumbing-valid ellipse masks and are not a final Gate16
  protocol.
- Official 49F VideoPainter Probe4 inference is still pending a free allowed
  GPU; no Gate16/Gate64 generation or training has started.

## 2026-06-23 Probe4 Official 49F Inference

- Status: `PROBE4_OFFICIAL_49F_INFERENCE_PASSED`.
- GPU: `2`.
- Rows: `4/4`.
- Frames: `49` for every row.
- Output:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623/exp26_probe4_official_inference`.
- Summary:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623/exp26_probe4_official_inference/probe4_official_inference_summary.json`.
- Report: `reports/exp26_probe4_official_inference_result.md`.
- Gate16/Gate64 and DPO training remain unstarted.

## 2026-06-24 Probe4 and Gate16 Review

- Status: `PROBE4_PASSED_GATE16_REVIEW_FAILED`.
- Controller run:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260624_three_lane_retry2`.
- Probe4: `4/4` official 49F inference outputs passed review.
- Gate16 source materialization: `25/32` valid formal 49F candidates; selected `16/16`.
- Gate16 official inference: `16/16`, all `49` frames.
- Gate16 review: `15/16` passed, `1/16` failed.
- Failed sample: `vp2_gate16_BLENDER_CON001_00742`, mask PSNR `1.3500`.
- Gate64: not launched.
- Report: `reports/exp26_probe4_gate16_review_20260624.md`.

## 2026-06-24 Gate16 Reclassification

- Status:
  - `PROBE4_PASSED`
  - `GATE16_15_OF_16`
  - `GATE16_METRIC_GATE_PASS_VISUAL_REVIEW_PENDING`
  - `NO_GATE64`
  - `NO_DPO_TRAINING`
- Existing Gate16 rows reclassified: `16/16`.
- Classification: `medium-hard=15`, `trivial-bad=1`, `technical-invalid=0`.
- Failed row retained as true model failure: `vp2_gate16_BLENDER_CON001_00742`.
- Original numeric gate criteria pass, but interactive video playback is still pending, so `GATE16_PASSED_WITH_REJECTION` is not set.
- Report: `reports/exp26_gate16_reclassification.md`.
