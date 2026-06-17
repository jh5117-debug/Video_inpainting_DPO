# Registry: Exp16 Prior-Confidence Gated DPO

Exp16 is a new mainline attempt after Exp11 outer b0.75 S2. It is not OR, not
adapter training, and not adaptive normalization.

Current state: Stage1 500 limit=100 engineering gate completed. This confirms
real prior-cache loading, latent-x0 loss wiring, preflight, checkpointing, and
dpo_diag writing. It is not a final metric result.
