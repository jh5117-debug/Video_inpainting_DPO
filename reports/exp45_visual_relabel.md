# Exp44 Targeted Visual Relabel

- Status: MINIMAX_TARGETED_RELABEL_COMPLETED
- Review pages opened: 8
- Candidates relabeled: 72
- SUCCESS_CLEAN: 8
- SUCCESS_USABLE including clean: 28
- FAILURE_MEDIUM_HARD: 22
- Rejected/borderline: 22
- Same-source groups with both success and failure: 4
- One-to-one same-source pair precheck: 8
- Capped same-source combination precheck: 16

## Visual Findings

- Clean success is concentrated in simple indoor, snow, stair/hallway, and stable-road scenes.
- Usable success often has minor geometry/reflection or boundary uncertainty and should be treated as pseudo-success, not GT.
- Medium-hard failures are mostly bounded residuals, local smears, water/reflection misses, or mild geometry errors with outside preservation intact.
- Rejected candidates include fogging/over-erasure, too-close rows, boundary destruction, and outside-damaged rows.

## Gate

The relabel pass is complete and unlocks Exp44 Milestone D pair construction. Pair construction must still enforce same-source pairing, scene split disjointness, and the >=24 usable pair gate before any later bad-noise or handoff step.
