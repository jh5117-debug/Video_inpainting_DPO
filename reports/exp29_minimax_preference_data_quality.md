# Exp29 MiniMax Preference Data Quality Gate

Date: 2026-06-26

Status: `MINIMAX_DATA_YIELD_INSUFFICIENT`

## Scope

This gate tested whether MiniMax-Remover can provide enough medium-hard
preference data for a bounded optimizer/precision micro gate. It did not run
any adapter training. Left CLI runtime and worktrees were not modified.

## Inputs

- Source pool: 32 non-VOR-Eval rows selected from the existing Exp25 VOR triplet
  semantic audit.
- Frames: 17 per source for the micro data gate.
- Seeds per source: `20260626`, `20260627`, `20260628`.
- Candidate generation: MiniMax official inference, raw output only, no hard
  comp, fixed source/mask/winner frame mapping.
- Candidate output root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/minimax_micro_data_quality_20260626/candidate_generation_17f`
- Source manifest:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/minimax_micro_data_quality_20260626/source32_17f/minimax_micro_source32.jsonl`
- Source manifest SHA256:
  `71232474115859bfa71ece4a7b189416043bf641217425604b5309941af3514a`

## Generation Result

- Candidates expected: 96
- Candidates completed: 96
- Worker GPUs: GPU0 and GPU5
- No right-side Exp29 GPU worker remains running.

## Classification

Candidate-level counts:

| Class | Count |
| --- | ---: |
| `MEDIUM_HARD_ELIGIBLE` | 23 |
| `HARD_BUT_PLAUSIBLE` | 4 |
| `TOO_CLOSE` | 3 |
| `TRIVIAL_BAD` | 60 |
| `TECHNICAL_INVALID` | 6 |

Eligible candidate rows: 27 / 96.

However, those 27 rows cover only 9 unique `sample_id` / scene groups because
the same source often produced similar outcomes across all three seeds. This is
not enough to build scene-disjoint `train16` and `heldout16`.

## Video Review

The review evidence consists of 24 page images, four candidates per page, each
showing 16 temporal samples. Codex opened and inspected all 24 pages.

Observed pattern:

- Several human/indoor cases are plausible medium-hard candidates, with local
  red-object remnants or incomplete removal that could be useful as losers.
- Many toy/object cases are trivial-bad: the target object remains visibly
  present, the removal is globally implausible, or the local artifact is too
  obvious.
- Some candidates are technical-invalid because masks or frame evidence are not
  suitable for a meaningful OR preference.
- Seed diversity is weak: when a source works, all three seeds tend to work; when
  it fails, all three tend to fail. Counting seeds as independent examples would
  overstate data diversity.

## Decision

`MINIMAX_MICRO_DATA_READY` is not granted.

The next MiniMax optimizer/precision recipe gate and 30-step confirmatory micro
gate are blocked until a larger or different source pool yields at least 16
train and 16 heldout scene-disjoint medium-hard / hard-plausible rows.

This result does not invalidate MiniMax as a future adapter candidate. It says
the current 32-source, 3-seed gate is not enough for a quality-positive
micro-training claim.

## Artifacts

- `reports/exp29_minimax_preference_data_quality.csv`
- `reports/exp29_minimax_preference_data_quality_summary.json`
- `reports/exp29_minimax_micro_review_pages/`
- `exp29_or_adapter_feasibility/manifests/minimax_micro_train16.jsonl`
- `exp29_or_adapter_feasibility/manifests/minimax_micro_heldout16.jsonl`
- `exp29_or_adapter_feasibility/manifests/minimax_micro_rejected.jsonl`
