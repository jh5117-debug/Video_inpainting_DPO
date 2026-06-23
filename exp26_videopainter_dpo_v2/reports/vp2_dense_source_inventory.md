# Exp26 Dense Source Inventory

Date: 2026-06-23

## Sparse YouTube-VOS

Path:

`/mnt/workspace/hj/nas_hj/data/external/ytbv_2019_full_resolution/train`

Result:

- valid 49F candidates: `0`
- failed candidates: `3471`
- max frame count seen: `36`
- max mask count seen: `36`

Decision: not usable for formal VideoPainter 49F preference data.

## VOR-Train BG Fallback

The formal requirement is 49 real clean frames plus generated or available BR
masks. It does not require 49 original object annotation masks. VOR-Train BG
videos are valid clean-video candidates if scene groups are isolated and
VOR-Eval/search/shadow groups are excluded.

Current source split created:

- train source: `128`
- search-dev: `32`
- shadow-dev: `32`
- candidate rows after exclusions: `51681`
- unique scene groups after exclusions: `1380`
- train/search group overlap: `0`
- train/shadow group overlap: `0`
- search/shadow group overlap: `0`
- status: `VOR_BG_SOURCE_SPLIT_LOCKED_PENDING_EXTRACTION_MASKS`
- train SHA256: `68a862c18ad46271757d96c6b5483c76aeb2cfdfb5cae21a452e215b735daaa6`
- search SHA256: `41c6571d26e4a5130818dd50fbbe1314c9d953284561a3cd20f630572f7c2a71`
- shadow SHA256: `0338dba1513cfe0e5dd85cbf793b3782902b981ac9610b0e472c6a048f738c02`

This is a first formal fallback split, not a generated loser manifest. Gate64
VideoPainter generation is still pending extraction, moving BR mask generation,
and official baseline inference.

Existing extracted VOR-Train Gate128 BG videos were sampled as source-format
evidence. Twelve decoded videos all had 240 real frames at 24 fps and
1920x1080 resolution. This does not replace exact probe of the locked Exp26
source split.

## Decision

Use VOR-Train BG as the current fallback source path for Exp26 formal 49F
construction. Record explicitly in PRD as `VOR-BG-used-as-clean-BR-source`.
