# Exp55 Failure Pattern Analysis

Status: `EXP55_FAILURE_PATTERN_IDENTIFIED`

The cross-lane evidence points to an objective/region-allocation problem, not a cache or runtime failure.

- Cache/runtime blocker is fixed: Exp53B produced checkpoints, strict reloads, and heldout videos.
- Outside preservation is generally safe or improved.
- Object/mask can improve, especially H20 R1/R2.
- Overlap / affected / boundary regressions remain the common blocker.
- Visual review is mixed/unsafe; no candidate reaches the original visual gate.
- R1 is safer than R2/R3/R4, but still misses boundary/overlap/affected gates.
- R2 loser clipping did not help enough; R2 has stronger local spill than R1.
- Exp54 R4_Q2_T500 has good full/outside/affected diagnostics, but object/boundary/visual are worse than H20 R1_Q2_T500.

Conclusion: current objectives over-optimize object/global preservation while under-protecting transition regions.
