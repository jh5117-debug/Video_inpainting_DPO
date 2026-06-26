# Exp29 Status

Current status: `READBACK_AND_SCAFFOLD_CREATED`

Exp29 is an isolated feasibility audit for MiniMax-Remover and EffectErase.
It inherits the Exp26 conclusion that DiffuEraser plus VideoPainter support
`CROSS_BACKBONE_ADAPTER_EVIDENCE_CONFIRMED`, but not a universal-adapter
claim.

No GPU inference, trainable-forward gate, DPO step, or long training has been
launched by the scaffold milestone.

## 2026-06-26 Repo And Weight Audit

- MiniMax: `MINIMAX_REPO_READY`, `MINIMAX_WEIGHTS_READY`.
- EffectErase: `EFFECTERASE_REPO_READY`, `EFFECTERASE_BLOCKED_NO_WEIGHTS`.
- No inference smoke or trainable-forward gate has run yet.

Reports:

- `reports/exp29_minimax_repo_weight_audit.md`
- `reports/exp29_minimax_repo_weight_audit.csv`
- `reports/exp29_minimax_asset_matrix.json`
- `reports/exp29_effecterase_repo_weight_audit.md`
- `reports/exp29_effecterase_repo_weight_audit.csv`
- `reports/exp29_effecterase_asset_matrix.json`

## 2026-06-26 Inference Smoke And Trainable Forward

- MiniMax inference: `MINIMAX_INFERENCE_SMOKE_PASSED_WITH_VISUAL_QUALITY_RISKS`.
- MiniMax visual quality: mixed; one medium-hard candidate and three
  trivial-bad outputs.
- MiniMax trainable forward: `MINIMAX_TRAINABLE_FORWARD_PASSED`.
- EffectErase inference: `EFFECTERASE_INFERENCE_SMOKE_BLOCKED_NO_WEIGHTS`.

## 2026-06-26 MiniMax Adapter Gates

- Zero-gap: `MINIMAX_ZERO_GAP_PASSED`.
- One-step strict reload: `MINIMAX_ONE_STEP_STRICT_RELOAD_PASSED`.
- 10-step: `MINIMAX_10STEP_PARETO_MIXED`.
- Heldout visual result: Step10 is nearly unchanged from Step0 on two heldout
  rows; no visible quality gain.
- MiniMax final: `MINIMAX_ADAPTER_POSSIBLE_NEEDS_MORE_WORK`.
- EffectErase final for this run: `EFFECTERASE_BLOCKED`.

## 2026-06-26 Continuation Readback

- Status: `EXP29_CONTINUATION_READBACK_COMPLETED`.
- Branch/HEAD confirmed:
  `research/exp29-minimax-effecterase-adapter-feasibility-20260626` at
  `4b8d68af3ebd0f6981e697baee952b5f0e1ca76f`.
- Previous PRD, registry, reports, MiniMax gate JSON, EffectErase asset matrix,
  and relevant code pointers were reread before any GPU task.
- Left CLI state was checked read-only. GPU1/GPU2/GPU3/GPU4 remain reserved by
  CLI runtime locks; Exp28 DAVIS50 evaluation was observed on GPU3.
- No left CLI signal was sent; no left runtime/worktree/output file was
  modified.
- Report: `reports/exp29_continuation_readback.md`.

## 2026-06-26 MiniMax 10-Step Failure Analysis

- Status: `MINIMAX_10STEP_FAILURE_ANALYZED`.
- Root cause: the previous stable recovery recipe was intentionally too
  conservative (`SGD(lr=1e-7)`), producing only
  `1.1061271569642785e-10` step10 parameter-probe delta.
- Backward path was not missing: mean preclip grad norm was `0.7237282794`, max
  `1.2341757971`, with 461 gradient tensors.
- Quality signal was weak: 3/4 previous smoke training losers were
  trivial-bad, and the heldout set had only 2 rows.
- Decision: do not run longer from the same recipe; require a medium-hard
  train16/heldout16 data-quality gate before further MiniMax optimizer tests.
- Reports:
  - `reports/exp29_minimax_10step_failure_analysis.md`
  - `reports/exp29_minimax_10step_failure_analysis.csv`
  - `reports/exp29_minimax_next_micro_plan.md`

## 2026-06-26 MiniMax Preference Data Quality Gate

- Status: `MINIMAX_DATA_YIELD_INSUFFICIENT`.
- Candidate protocol: 32 sources, 17 frames each, 3 fixed seeds each, raw
  MiniMax output only.
- Reviewed candidates: 96.
- Counts:
  - `MEDIUM_HARD_ELIGIBLE`: 23
  - `HARD_BUT_PLAUSIBLE`: 4
  - `TOO_CLOSE`: 3
  - `TRIVIAL_BAD`: 60
  - `TECHNICAL_INVALID`: 6
- Eligible candidates: 27, but only 9 unique scene groups.
- The required scene-disjoint train16/heldout16 split cannot be formed.
- Decision: recipe search and 30-step MiniMax micro are not allowed from this
  data gate.
- Reports:
  - `reports/exp29_minimax_preference_data_quality.md`
  - `reports/exp29_minimax_preference_data_quality.csv`
  - `reports/exp29_minimax_preference_video_review.csv`
  - `reports/exp29_minimax_preference_data_quality_summary.json`

## 2026-06-26 EffectErase Weight Recovery

- Status: `EFFECTERASE_WEIGHTS_READY`.
- Official EffectErase and Wan2.1-Fun InP assets recovered into the Exp29
  autoresearch cache.
- SHA256 verification: 19 manifest entries checked, all `OK`.
- No inference smoke, trainable forward, zero-gap, one-step, or adapter micro
  has run yet.
- EffectErase remains VOR baseline/diagnostic only until a non-confounded smoke
  and evaluation design are completed.
