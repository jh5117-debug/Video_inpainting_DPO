# Exp35 Registry

Exp35 tracks the MiniMax flow-DPO rescue after Exp30 showed data readiness and
technical plumbing success but no quality-positive 10-step output movement.

Current status: `MINIMAX_RESCUE_RECIPES_PREREGISTERED`.

Gate order:

1. Readback and scaffold.
2. No-change forensic audit.
3. Inference sensitivity positive-control.
4. Trainable-scope audit and minimal expansion if needed.
5. Winner-SFT positive-control.
6. Bad-noise/hard-timestep miner.
7. Bounded 10-step rescue recipe gate.
8. Conditional 30-step only after 10-step recipe pass.

No universal-adapter, final-SOTA, RC-FPO, or long-training claims are allowed.

Latest milestone:

- Bad-noise / hard-timestep miner scanned 32 train and 16 heldout rows with
  16 candidate states per row, wrote fixed state manifests, and launched no
  training. This prepares bounded 10-step recipe testing only; 30-step remains
  locked.
- Rescue recipe preregistration locked three active 10-step recipes and no
  training. R4 SDPO-safe hybrid remains inactive until MiniMax SDPO true-model
  parity exists.
