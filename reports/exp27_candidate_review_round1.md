# Exp27 Candidate Review Round 1

Round 1 consensus:

- C1 and C2 are the strongest. They are task-native and inpainting-specific, which is the safest novelty boundary after LocalDPO.
- C4 is promising but heavier; it should become a fallback or second-stage story after data/region evidence is solid.
- C3 and C5 should not be primary methods. They are necessary baselines/diagnostics against SDPO and Linear-DPO.

Action after Round 1:

1. Merge C1 and C2 into the primary method family: task-native, failure-structured preference data plus restoration-critical region decomposition.
2. Keep C4 as fallback if data/region evidence is insufficient but stage-specific failures remain clear.
3. Treat C3/C5 as exact objective baselines and optional ablations, not novelty.
