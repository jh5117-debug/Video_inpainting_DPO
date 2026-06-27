# Exp30 Controlled-Corruption Smoke16 V3

Status: `CONTROLLED_CORRUPTION_V3_READY`

- Candidate count: 24
- Source count: 16
- All classification counts: `{'TRIVIAL_BAD': 8, 'MEDIUM_HARD_ELIGIBLE': 16}`
- Primary classification counts: `{'TRIVIAL_BAD': 3, 'MEDIUM_HARD_ELIGIBLE': 13}`
- Primary usable count: 13
- Primary technical-valid count: 16
- Primary outside-fail count: 0

The v3 controlled fallback follows the preregistered profile schedule from `exp30_controlled_corruption_v3_plan.json`: CC-v3-B for all sources, CC-v3-A on six repair sources, and CC-v3-C on two affected soft sources.  It is still only one component of Smoke16 v3 and does not unlock Gate64 or adapter training by itself.

## Codex Visual Review

- Review status: `24/24` all controlled v3 candidates reviewed via temporal
  evidence pages.
- Primary review status: `16/16` primary controlled candidates reviewed.
- Pages opened: 6 all-candidate pages and 4 primary pages.
- Visual conclusion: v3 substantially reduces the v2 frame-wise temporal jump
  pattern and preserves outside regions. Three primary candidates remain
  visibly over-hard / temporally unstable and are kept as `TRIVIAL_BAD`.
- Final primary visual counts: `13` medium-hard, `0` hard-plausible, `0`
  too-close, `3` trivial-bad, `0` technical-invalid.
- This subgate is `CONTROLLED_CORRUPTION_V3_READY`, but full Smoke16 v3 still
  requires the multi-model aggregate with MiniMax, DiffuEraser, and ProPainter.
