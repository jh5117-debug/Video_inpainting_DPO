# Exp25 DiffuEraser Primary OR Stack Decision

Date: 2026-06-23

## Current Decision

No DiffuEraser OR primary generator is locked yet.

Status:

`PRIMARY_STACK_PENDING`

Reason:

- The official OR route uses PCM 2-step inference acceleration.
- The current DPO/PAI environment fails while loading the PCM LoRA acceleration weights.
- Silently disabling PCM would change the generator identity and is disallowed.
- The historically verified no-PCM protocol is BR/DAVIS evaluation, not yet a verified OR/VOR generator.

## Candidate Stacks

1. `DE_OFFICIAL_PCM2`
   - Preferred official baseline candidate.
   - Requires official-pinned environment smoke.
   - Current active env failed.

2. `DE_CANONICAL_NO_PCM`
   - Policy-matched diagnostic candidate.
   - Must be explicitly named `diffueraser_or_no_pcm_<config_hash>`.
   - Requires an OR smoke; cannot inherit BR-only verification.

## Gate Before Gate128

Gate128 is blocked until:

- one DiffuEraser stack is strict-load verified;
- six fixed smoke samples run without fallback;
- raw/no-comp outputs are decoded and checked;
- visual review is completed;
- generator identity is written into the manifest.

Until then, Exp25 remains:

`GATE128_EXTRACTED`

`THREE_MODEL_SMOKE_PARTIAL_BLOCKED`
