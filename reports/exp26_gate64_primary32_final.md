# Exp26 Gate64 Primary-32 Final Manifest

Status: `GATE64_DATA_READY`

Primary manifest: `exp26_videopainter_dpo_v2/manifests/vp2_gate64_primary32_final.jsonl`
SHA256: `82f661f2f30a581a213972533817624217eabb97eba7aaeedc00ee2109e4e716`

Reserve manifest: `exp26_videopainter_dpo_v2/manifests/vp2_gate64_reserve_final.jsonl`
SHA256: `84d3ede678357c2fc3e61d84d49922b4668665ba9fa29222fa1fac4c82ac3e38`

Rejected manifest: `exp26_videopainter_dpo_v2/manifests/vp2_gate64_rejected_final.jsonl`
SHA256: `aadea5949bd63170986e8cdd3c255a1ca50c033f593059074a3f5da242c68a8d`

Final counts:
- Gate64 formal-valid: 64/64
- Eligible: 55
- Rejected: 9
- Primary rows: 32
- Reserve rows: 23
- Search-dev scene overlap: 0
- Shadow-dev scene overlap: 0
- VOR-Eval overlap: not used; Gate64 comes only from locked VOR-Train-BG source split.

Primary balance:

| dimension | distribution |
| --- | --- |
| classification | `{'hard-plausible': 16, 'medium-hard': 16}` |
| area | `{'small': 8, 'large': 8, 'medium': 16}` |
| motion | `{'low': 8, 'high': 8, 'medium': 16}` |
| mask profile | `{'irregular_freeform': 8, 'object_like_polygon': 8, 'soft_blob': 5, 'ellipse_circle_subset': 3, 'thin_structure_freeform': 5, 'edge_touch_freeform': 3}` |
| loser | `{'comp': 32}` |

Loser semantics:
- Training uses `final_loser_video_path`, and every primary row points to the comp loser.
- `train_videopainter_dpo_adapter.py` resolves loser as `final_loser_video_path` -> `comp_loser_video_path` -> `raw_loser_video_path`.
- Raw outputs are retained only for outside-damage diagnostics and ablation.

Decision:
- The existing primary-32 draft remains valid after final temporal review.
- This allows VP-L0/L1 and later 10-step micro-training to start.
