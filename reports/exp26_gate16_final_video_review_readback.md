# Exp26 Gate16 Final Video Review Readback

Date: 2026-06-24

## Branch / HEAD

- Branch: `research/exp26-videopainter-dpo-v2`
- HEAD: `57461ad89c3fed1fefc2a0680516febc6165a517`
- Remote fetch: completed with `git fetch --all --prune`
- Worktree status before milestone: clean

## Files Read

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/48_exp26_videopainter_dpo_v2.md`
- `experiment_registry/exp26_videopainter_dpo_v2/status.md`
- `reports/exp26_probe4_gate16_review_20260624.md`
- `reports/exp26_gate16_reclassification.md`

## Source Of Truth For This Milestone

- Existing Gate16 inference output only; no replacement samples.
- PAI run root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260624_three_lane_retry2`
- Gate16 inference: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260624_three_lane_retry2/exp26_gate16_official_inference`
- Existing reclassification root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_26_27_followup_20260624/exp26_gate16_reclassification_v2`

## Completed Work Not To Repeat

- L0-L4 static/plumbing gates are already recorded as passed.
- Probe4 official 49F inference passed.
- Gate16 generated `16/16` outputs with `49` frames each.
- Existing metric reclassification found `15` medium-hard and `1` trivial-bad true model failure.

## Banned Repeats / Banned Promotions

- Do not replace `vp2_gate16_BLENDER_CON001_00742`.
- Do not launch Gate64 before final visual review satisfies the hard rule.
- Do not start VideoPainter DPO training.
- Do not mark `VIDEO_REVIEW_PASS` unless every sample has an opened review pack and final row-level judgement.

## Promotion Gate

Pre-registered Gate16 pass requires:

- technical valid >= 15/16;
- systematic failure = 0;
- trivial-bad <= 2/16;
- medium-hard or hard-plausible >= 8/16;
- final visual review complete.

This milestone only attempts to satisfy the final visual review requirement using dense per-sample review packs, because interactive mp4 playback is not available in the current Codex UI.
