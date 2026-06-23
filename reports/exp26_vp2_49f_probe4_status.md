# Exp26 VOR-BG 49F Probe4 Status

Date: 2026-06-23

Runtime snapshot:

`/mnt/workspace/hj/nas_hj/runtime_code_snapshots/exp26_7f9ec40`

Source:

`exp26_videopainter_dpo_v2/manifests/vp2_vor_bg_train_source_128.jsonl`

Probe scope:

- 4 train-source VOR-BG rows.
- CPU/NAS selective extraction only; no VideoPainter inference and no GPU training.
- VOR-Eval was not used.

Selective extraction:

- Extractor: Exp25 safe selective VOR extraction utility.
- Archive root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/downloads/fa09dc61128ca0418a4a13364d97a08018ea9cc7`
- Output root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/extracted/exp26_vp2_49f_probe4`
- State:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/49f_probe_7f9ec40/selective_extract_probe4_state.json`
- Result: `4/4` target BG mp4 files written, `unsafe=0`.
- The archive stream scanned `38,314` members before all four targets were found.

Formal 49F materialization:

- Output manifest:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/49f_probe_7f9ec40/vp2_probe4_49f_materialized.jsonl`
- Result: `4/4` decoded with exactly `49` unique real frames.
- Source videos: all `240` frames, `30.0` fps, `1280x720`.
- No padding, looping, interpolation, duplicated-frame fallback, or 13-frame plumbing mode was used.

Moving BR masks:

- Output manifest:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/49f_probe_7f9ec40/vp2_probe4_49f_masks.jsonl`
- Result: `4/4` masks generated.
- `first_frame_gt=true`: all four rows have `first_frame_sum=0`.
- Area means range from `0.0669` to `0.2066`.
- Centroid motion ranges from `142.17px` to `302.90px`.

Decision:

`FORMAL_49F_PROBE4_PASSED`

Gate64 is still not started. The next safe step is to run the same extraction
and mask-generation path for the locked Gate64 source subset, then perform
VideoPainter official self-loser inference and visual review.
