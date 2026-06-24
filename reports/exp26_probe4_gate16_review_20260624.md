# Exp26 Probe4 and Gate16 Review - 2026-06-24

Status: `PROBE4_PASSED_GATE16_REVIEW_FAILED`

Controller run:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260624_three_lane_retry2`

Probe4:

- Official VideoPainter 49-frame inference: `4/4` completed.
- Review: `4/4` passed.
- Mean PSNR: `20.3849`.
- Mean mask PSNR: `12.6602`.
- Mean temporal absdiff: `5.3371`.
- Black frame ratio: `0.0`.
- Visual review: all four contact sheets decode correctly with no frame-count or black-frame issue.

Gate16:

- Source pool: first 32 Gate32 rows were materialized to locate formal 49-frame inputs.
- Materialization: `25/32` valid formal 49-frame sources, `7/32` rejected.
- Selected Gate16 manifest: `16/16` valid formal 49-frame rows.
- Mask generation: `16/16`.
- Official VideoPainter 49-frame inference: `16/16` completed, `49` frames each.
- Review: `15/16` passed, `1/16` failed.

Gate16 aggregate diagnostics:

| metric | mean | min | max |
| --- | ---: | ---: | ---: |
| PSNR | 20.5761 | 8.7972 | 27.4897 |
| mask PSNR | 12.6980 | 1.3500 | 19.0344 |
| temporal absdiff | 4.2682 | 1.2341 | 14.2906 |
| black frame ratio | 0.0000 | 0.0000 | 0.0000 |

Failed Gate16 row:

- sample: `vp2_gate16_BLENDER_CON001_00742`
- frames: `49`
- PSNR: `8.7972`
- mask PSNR: `1.3500`
- temporal absdiff: `2.5925`
- black frame ratio: `0.0`
- visual reason: masked region becomes a large mismatched pale object/patch rather than a plausible inpainted background.

Outputs:

- Probe4 review: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260624_three_lane_retry2/exp26_probe4_review`
- Gate16 selected manifest: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260624_three_lane_retry2/exp26_gate16_selected16_49f_materialized.jsonl`
- Gate16 inference: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260624_three_lane_retry2/exp26_gate16_official_inference`
- Gate16 review: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260624_three_lane_retry2/exp26_gate16_review`
- Gate16 visual index: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260624_three_lane_retry2/exp26_gate16_contact_sheet_index.jpg`

Decision:

Gate16 did not pass because one of sixteen samples failed the quantitative and visual review. Gate64 was not launched, and no VideoPainter DPO training was started.
