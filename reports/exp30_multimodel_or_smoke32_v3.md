# Exp30 Multi-Model OR Smoke32 V3 Aggregate

Status: `MULTIMODEL_OR_SMOKE32_V3_PASS`

- Candidate rows: 64
- Source rows: 16
- Technical-valid candidates: 64/64 (1.000)
- Total usable candidates: 14
- Best-per-source usable: 10/16
- Controlled usable source coverage: 8/16
- Usable generator families: `['controlled_corruption_v3', 'minimax_official_v3', 'propainter']`
- Best-per-source counts: `{'HARD_BUT_PLAUSIBLE': 1, 'MEDIUM_HARD_ELIGIBLE': 9, 'TRIVIAL_BAD': 6}`
- Model classification counts: `{'controlled_corruption_v3:MEDIUM_HARD_ELIGIBLE': 8, 'controlled_corruption_v3:TRIVIAL_BAD': 8, 'diffueraser:TRIVIAL_BAD': 16, 'minimax_official_v3:HARD_BUT_PLAUSIBLE': 1, 'minimax_official_v3:MEDIUM_HARD_ELIGIBLE': 2, 'minimax_official_v3:TOO_CLOSE': 1, 'minimax_official_v3:TRIVIAL_BAD': 12, 'propainter:HARD_BUT_PLAUSIBLE': 1, 'propainter:MEDIUM_HARD_ELIGIBLE': 2, 'propainter:TRIVIAL_BAD': 13}`

Smoke32 does not launch Gate64 or training by itself; it only decides whether the limited Gate64 pool may be prepared next.
