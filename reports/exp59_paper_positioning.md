# Exp59 Paper Positioning

Exp59 should be positioned as a diagnostic, not as adapter evidence.

## Safe Claim

VOID official inference can run on the project-generated Kubric-native Gate8 format. The run produced technically valid outputs for 8/8 samples and preserved outside/background regions in visual review.

## Limitation

The Exp58B Kubric Gate8 data is target-hit weak: all 8 rows have `target_hit=false`. The official inference outputs are therefore not sufficient to confirm data-mismatch relief or adapter readiness.

## Paper Role

VOID remains useful as:

- an inference baseline,
- a same-model loser generator candidate,
- an adapter-engineering diagnostic.

VOID should not be described as:

- a third adapter backbone,
- a universal adapter result,
- a final SOTA result,
- a positive adapter result.

## Recommended Wording

“For VOID, official inference and native Kubric-format data generation were validated, but the first Kubric Gate8 carried a target-hit weakness and did not resolve the transition-region failure pattern. We therefore keep VOID as a diagnostic baseline and loser-generator candidate pending target-hit-positive native data.”
