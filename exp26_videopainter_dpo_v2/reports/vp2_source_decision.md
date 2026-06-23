# Exp26 VideoPainter 49F Source Decision

Date: 2026-06-23

Status:

`VOR_BG_SOURCE_SPLIT_LOCKED_PENDING_EXTRACTION_MASKS`

Sparse YouTube-VOS remains invalid for formal 49-frame VideoPainter preference data: previous audit found `0` valid 49F candidates and maximum observed frame count `36`.

The corrected formal requirement is:

- exactly 49 real continuous frames;
- a clean winner video;
- a generated or available 49-frame training mask.

It is not required that the source already contain 49 original annotation masks.

## Selected Fallback

`VOR-BG-used-as-clean-BR-source`

Input index:

`/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp25_vor/exp25_vor_or_preference_data/manifests/vor_train_metadata_index.jsonl`

The index was copied temporarily from the completed Exp25/PAI VOR-Train metadata index for split construction. The 23MB full metadata index is not committed; only the source split manifests and statistics are committed.

Exclusions:

- Exp25 search-dev.
- Exp25 shadow-dev.
- Exp25 Gate128.
- VOR-Eval is excluded by source construction.

Selection rule:

- one representative per `scene_group`;
- split by unique scene group;
- no train/search/shadow scene overlap;
- source-only rows only.

Counts:

- rows after exclusions: 51,681
- unique scene groups after exclusions: 1,380
- train source: 128
- search-dev source: 32
- shadow-dev source: 32
- train/search overlap: 0
- train/shadow overlap: 0
- search/shadow overlap: 0

Manifest SHA256:

- train: `68a862c18ad46271757d96c6b5483c76aeb2cfdfb5cae21a452e215b735daaa6`
- search: `41c6571d26e4a5130818dd50fbbe1314c9d953284561a3cd20f630572f7c2a71`
- shadow: `0338dba1513cfe0e5dd85cbf793b3782902b981ac9610b0e472c6a048f738c02`

## Frame Evidence

Existing extracted VOR-Train Gate128 BG videos were probed only as format evidence. Twelve sampled BG videos decoded as 240 frames, 24 fps, 1920x1080.

This supports VOR-Train BG as a likely 49F-capable source, but selected Exp26 source videos still require selective extraction and exact decode audit before Gate64.

## Gate64 Status

Gate64 has not started.

Remaining blockers:

1. Selectively extract selected VOR-BG source videos.
2. Confirm exact 49-frame mapping from real decoded frames.
3. Generate moving BR masks.
4. Run official VideoPainter baseline to create Gate64 self-losers.
5. Perform quantitative and visual quality audit.

No 13F fallback is allowed for formal Gate64.
