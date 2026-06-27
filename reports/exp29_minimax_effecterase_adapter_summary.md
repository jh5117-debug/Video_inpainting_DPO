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

Final status: `EFFECTERASE_OR_BASELINE_READY`

EffectErase now has verified official weights and Wan2.1-Fun InP assets. The
official 81-frame inference smoke ran 8/8 diagnostic VOR rows successfully.
Codex opened all temporal review pages and crop pages. The outputs show strong
object/effect removal with no black/purple collapse.

This result promotes EffectErase to OR strong baseline / diagnostic status. It
does not promote EffectErase to true adapter status because no training forward,
zero-gap, one-step, or non-confounded heldout adapter gate has run.

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

EffectErase provides a strong OR baseline/diagnostic path, but it has not
provided true-adapter evidence.
