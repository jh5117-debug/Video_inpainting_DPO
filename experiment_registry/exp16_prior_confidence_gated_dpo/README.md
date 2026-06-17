# Registry: Exp16 Prior-Confidence Gated DPO

Exp16 is a new mainline attempt after Exp11 outer b0.75 S2. It is not OR, not
adapter training, and not adaptive normalization.

Current state: Stage1 500 limit=100 engineering gate plus DAVIS10 visual sanity
completed. This confirms real prior-cache loading, latent-x0 loss wiring,
preflight, checkpointing, dpo_diag writing, and eval loading through a
DPO-S1 + SFT-S2 hybrid checkpoint.

DAVIS10 shows weak positive signal over SFT-48000, but Exp16 does not beat the
current best Exp11 outer b0.75 S2. It is not a final method result and should
not be expanded to full cache/full training without another small gate.
