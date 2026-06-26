# Exp29 MiniMax / EffectErase Adapter Summary

Date: 2026-06-26

## MiniMax

Final status: `MINIMAX_ADAPTER_POSSIBLE_NEEDS_MORE_WORK`

MiniMax is the stronger true-adapter candidate in this round. It has:

- verified local official repo;
- verified NAS weights;
- official-style inference smoke;
- native flow target audit;
- differentiable trainable forward;
- zero-gap pass;
- one-step strict reload pass;
- 10-step technical micro completion after using a conservative non-final
  optimizer.

The blocker is quality, not basic plumbing. Inference smoke quality is mixed
and the 10-step heldout outputs are effectively unchanged from Step0.

## EffectErase

Final status: `EFFECTERASE_BLOCKED`

EffectErase has a local official repo but no verified official weights in the
audited paths. It remains a strong OR baseline / diagnostic candidate once
assets are available, but this run cannot validate inference or adapter gates.

EffectErase remains VOR-confounded for scientific adapter claims because the
model is trained on VOR. VOR use should be diagnostic/baseline only unless a
non-confounded external OR validation design is locked.

## Scientific Language

Allowed now:

`CROSS_BACKBONE_ADAPTER_EVIDENCE_CONFIRMED`

Still not allowed:

`UNIVERSAL_ADAPTER`

`THIRD_BACKBONE_ADAPTER_FEASIBILITY_CONFIRMED`

MiniMax provides a credible next-adapter path, but it has not yet produced a
heldout-positive 10-step micro result.

