# Exp26 Gate64 Source Repair Readback

- Timestamp: 2026-06-25T01:53:11.055356+00:00
- Branch: `research/exp26-videopainter-dpo-v2`
- HEAD: `36bee39e4ecac7243748f316e958d2e0e3c8db59`
- PAI host: `dsw-753014-85f54df947-bkp7h`
- PAI GPUs: 8 x L20X, all 0 MiB at readback

## Files Read

- `PRD/48_exp26_videopainter_dpo_v2.md`
- `experiment_registry/exp26_videopainter_dpo_v2/status.md`
- `reports/exp26_gate64_human_visual_review.csv`
- `reports/exp26_gate64_duplicate_source_deep_audit.md`
- `reports/exp26_gate64_repair_dense_visual_review.csv`
- `exp26_videopainter_dpo_v2/code/materialize_vp2_49f_sources.py`
- `exp26_videopainter_dpo_v2/code/train_videopainter_dpo_adapter.py`

## Already Completed

- Post-maintenance NAS persistence for Exp26 Gate64 artifacts.
- Deep duplicate-source audit for 8 static-pixel duplicate rows.
- Materializer guard repair allowing static pixel duplicates when decoded frame indices/timestamps are valid.
- Repair materialization, mask generation, official VideoPainter generation, and dense visual evidence for 8 rows.
- Manual local inspection of the 8 repair evidence/crop sheets.

## Pending / Banned Repeats

- Do not rerun Gate16.
- Do not regenerate the original 56 Gate64 outputs.
- Do not start Gate128 or long training.
- Do not start VideoPainter DPO until experiment output permissions are fixed and the primary manifest is consumed by the trainer.

## Promotion Gate

- Formal-valid source rows must be >=62/64: passed after repair, 64/64.
- Eligible medium-hard/hard-plausible rows must be >=32: passed, 55 eligible.
- Primary source/defect-balanced manifest must exist: passed, primary-32 written.
- DPO output root writable: failed; root ACL fix still needed.

## Summary JSON

```json
{
  "total_gate64_sources": 64,
  "formal_valid_sources_after_repair": 64,
  "reviewed_outputs": 64,
  "classification_counts": {
    "medium-hard": 37,
    "hard-plausible": 18,
    "trivial-bad": 8,
    "too-close": 1
  },
  "eligible_count": 55,
  "rejected_count": 9,
  "selection_decision_counts": {
    "ELIGIBLE_AFTER_VISUAL_REVIEW": 55,
    "REJECT_TRIVIAL_OR_TECHNICAL": 8,
    "REJECT_TOO_CLOSE": 1
  },
  "candidate_pool_jsonl": "/home/hj/H20_Video_inpainting_DPO_exp26_videopainter/exp26_videopainter_dpo_v2/manifests/vp2_gate64_candidate_pool55_visual_reviewed_comp.jsonl",
  "candidate_pool_sha256": "4bec06d78249b809ddf90a80eca900f8467ba3a2798c8ae9d03552d0f25ee5b8",
  "primary32_jsonl": "/home/hj/H20_Video_inpainting_DPO_exp26_videopainter/exp26_videopainter_dpo_v2/manifests/vp2_gate64_primary32_visual_reviewed_comp.jsonl",
  "primary32_sha256": "21c0c617698bbf69859f3850979d777c3af3e1908926bca43b69ede447df3cc8",
  "primary32_counts": {
    "classification": {
      "hard-plausible": 16,
      "medium-hard": 16
    },
    "area_bucket": {
      "small": 8,
      "large": 8,
      "medium": 16
    },
    "motion_bucket": {
      "low": 8,
      "high": 8,
      "medium": 16
    },
    "mask_profile": {
      "irregular_freeform": 8,
      "object_like_polygon": 8,
      "soft_blob": 5,
      "ellipse_circle_subset": 3,
      "thin_structure_freeform": 5,
      "edge_touch_freeform": 3
    }
  },
  "loser_semantics": "final_loser_path uses comp_loser_video_path to match current BR selected_primary_comp source of truth; raw_loser_video_path retained for diagnostics/ablation",
  "dpo_training_status": "NOT_STARTED",
  "dpo_blocker": "PAI hj cannot write /mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp26_videopainter_dpo_v2 until root ACL fix is applied"
}

```
