# Exp57 PAI Next Steps

Do not run 10-step from the PAI Exp57 lane.

A later Exp58 aggregator should compare H20 and PAI one-step evidence. Under the original one-step PASS criteria, PAI provides no 10-step-unlocking candidate.

PAI observations:

- `ATS_SDPO_Q2_T500_S0` improved full/outside/affected metrics but failed object, overlap, boundary, and visual review.
- `ATS_LINEAR_Q2_T500_S0` nearly preserved object but still regressed overlap and boundary, with visual worse on every heldout row.
- Backtracking selected full update scale `1.0`; no update was rejected.
- The adaptive safety check is still too weak relative to heldout transition damage.

VOID remains baseline / loser-generator / adapter-engineering candidate, not third-backbone evidence.
