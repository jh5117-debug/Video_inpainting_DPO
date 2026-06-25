# Exp26 Gate64 Final Temporal Review Readback

Date: 2026-06-25

Branch: `research/exp26-videopainter-dpo-v2`

HEAD at milestone start: `c42eb517c22bd2191f1f8fd34479bf639e2dfed4`

Read before action:

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/48_exp26_videopainter_dpo_v2.md`
- `experiment_registry/exp26_videopainter_dpo_v2/status.md`
- `reports/exp26_gate64_final_readiness.md`
- `reports/exp26_gate64_source_repair_readback.md`
- `reports/exp26_gate64_final_human_visual_review.csv`
- `exp26_videopainter_dpo_v2/code/train_videopainter_dpo_adapter.py`
- `exp26_videopainter_dpo_v2/code/review_gate64_official_outputs.py`

Already completed:

- Gate64 official VideoPainter generation for all 64 rows.
- Static-pixel duplicate repair for the 8 previously blocked rows.
- Dense evidence and crop-sheet review.
- Primary-32 draft manifest using comp loser semantics.
- PAI post-maintenance write permission recovery.

Pending at this milestone start:

- Strict temporal evidence review across all 64 Gate64 samples.
- Final primary-32 manifest lock.
- Path/frame/decode validation of primary-32.

Banned repeats:

- No Gate64 regeneration.
- No source replacement.
- No seed change.
- No DPO training before `GATE64_DATA_READY`.

Promotion gate:

- 64/64 formal-valid.
- 64/64 final temporal review evidence.
- Eligible rows >= 32.
- Primary rows = 32.
- Search-dev scene overlap = 0.
- Shadow-dev scene overlap = 0.
- All training paths exist and decode as 49-frame data.

Outputs:

- `reports/exp26_gate64_final_temporal_review.md`
- `reports/exp26_gate64_final_temporal_review.csv`
- `reports/exp26_gate64_primary32_final.md`
- `reports/exp26_gate64_primary32_final.csv`
- `reports/exp26_gate64_primary32_path_frame_validation.csv`
- `reports/exp26_gate64_manifest_identity.json`
- `exp26_videopainter_dpo_v2/manifests/vp2_gate64_primary32_final.jsonl`
- `exp26_videopainter_dpo_v2/manifests/vp2_gate64_reserve_final.jsonl`
- `exp26_videopainter_dpo_v2/manifests/vp2_gate64_rejected_final.jsonl`

PAI:

- Hostname: `dsw-753014-85f54df947-bkp7h`
- Permission state: recovered for Exp26 experiment and autoresearch roots.
- GPU use: none for this review/manifest lock milestone.
