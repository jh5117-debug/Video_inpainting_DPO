# Exp29 MiniMax Expanded Data Quality V2

Status: `MINIMAX_EXPANDED_DATA_YIELD_INSUFFICIENT`

## Inputs
- Full-VOR source audit manifest: `minimax_full_vor_source_candidates_v2.jsonl`
- First-pass source96 manifest SHA256: `267d03a2991894a47e26a14b698f6fd28423a726e6968890448cd460e5de1928`
- Final materialized source96 SHA256: `a8998902daa8e771afd111e798df017e4b64f5f21f5a43bf5fa6ef82aa4ce428`
- Conditional seed_b near-miss manifest SHA256: `1d45c60cdd54a28fe98373bd88d53b4cb277c649c292ad9a3e00c4aa718a6aad`

## Attempt Counts
- Seed A candidates: 96
- Seed B near-miss candidates: 32
- Unique scene groups attempted after best-candidate merge: 96
- Eligible unique scene groups: 26

## Classification Counts, All Attempts
```json
{
  "HARD_BUT_PLAUSIBLE": 2,
  "MEDIUM_HARD_ELIGIBLE": 24,
  "TECHNICAL_INVALID": 11,
  "TOO_CLOSE": 14,
  "TRIVIAL_BAD": 77
}
```

## Classification Counts By Seed
```json
{
  "seed_a": {
    "HARD_BUT_PLAUSIBLE": 2,
    "MEDIUM_HARD_ELIGIBLE": 23,
    "TECHNICAL_INVALID": 11,
    "TOO_CLOSE": 7,
    "TRIVIAL_BAD": 53
  },
  "seed_b": {
    "MEDIUM_HARD_ELIGIBLE": 1,
    "TOO_CLOSE": 7,
    "TRIVIAL_BAD": 24
  }
}
```

## Final Selection
- Train rows: 16
- Heldout rows: 0
- Train SHA256: `84d3b3ce06216a05ea005fb29a91fdd40a1e73b7b1cd2ab7a49bb3e311683c95`
- Heldout SHA256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

## Decision
The expanded first pass plus the pre-registered seed_b near-miss rescue still did not reach 32 scene-disjoint eligible MiniMax candidates. Per the V4 gate, MiniMax recipe/30-step/training remains stopped.

## Evidence
- Combined CSV: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/minimax_expanded_data_quality_v2_20260626/exp29_minimax_expanded_data_quality_v2.csv`
- Visual review CSV: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/minimax_expanded_data_quality_v2_20260626/exp29_minimax_expanded_video_review_v2.csv`
- Seed A review pages: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/minimax_expanded_data_quality_v2_20260626/review_seed_a_v2/review_pages`
- Seed B review pages: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/minimax_expanded_data_quality_v2_20260626/review_seed_b_v2/review_pages`

## Codex Visual Review

- Seed A review pages opened: 24/24.
- Seed B near-miss review pages opened: 8/8.
- The pages contain per-sample 17-frame temporal evidence and classification labels.
- Observed pattern: a limited number of plausible medium-hard local defects, many trivial-bad outputs, several too-close outputs, and 11 technical-invalid rows.
- The visual evidence supports the data-yield-insufficient decision and does not support MiniMax quality-positive, recipe, 30-step, or third-backbone-positive language.
