# Exp26 Third-Model Next Adapter Recommendation

Recommended next true-adapter target: `CoCoCo`, conditional on acquiring verified weights and reproducing one-batch native target parity.

Reason: among the audited non-completed models, CoCoCo is closest to the SD/UNet diffusion family already handled by DiffuEraser, has local model forward code, and should expose noise/timestep semantics more naturally than non-diffusion ProPainter or Wan/DiT systems. Its blocker is real: missing weights/dependencies and no released official training code.

Recommended next baseline/loser-generator target: `MiniMax-Remover`, conditional on weights/environment restoration. It is fast and OR-native, but should not be treated as a ready DPO adapter until its flow-matching training target and reference/policy parity are proven.

Do not train EffectErase as primary VOR on-policy evidence: it is trained on VOR and should remain a diagnostic/strong baseline until a non-confounded validation design exists.
