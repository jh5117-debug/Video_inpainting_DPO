# PRD 43: Exp21 Multibackbone VideoDPO BR Plumbing Smoke

Date: 2026-06-19

## Scope

Exp21 validates whether other diffusion/video inpainting backbones can support
future VideoDPO/Diff-DPO training plumbing. It is smoke-only and does not claim
quality.

## Status

```text
MATRIX_SCAFFOLD_READY
```

The initial compatibility matrix generator is implemented. Real backend smoke
must still be run per model before any backend can be marked ready.

## Rules

- Do not put other model weights into DiffuEraser trainer.
- Keep native VAE/scheduler/prediction target.
- Winner/loser share noise and timestep.
- Reference frozen.
- Adapter/LoRA grad must be non-zero.
- Save/reload must preserve outputs.

EffectErase VOR remains:

```text
WAITING_AUTH
```
