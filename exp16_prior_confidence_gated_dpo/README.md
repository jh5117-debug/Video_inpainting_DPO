# Exp16 Prior-Confidence Gated DPO

Exp16 is a new isolated experiment line that starts from the current best
DiffuEraser DPO setting:

```text
Exp11 boundary outer b0.75 S2
region-local normalized-gap clipped-loser-gap winner-anchored DPO
```

The new idea is to add a real ProPainter-prior confidence gate:

- reliable ProPainter-prior region: preserve / stay close to the prior;
- unreliable prior region: allow GT/context/preference-driven generation;
- outer boundary: keep the seam constrained.

Current state:

```text
Stage1 500 limit=100 engineering gate completed.
```

The limit=100 gate used real ProPainter prior frames, reconstructed predicted
latent x0, and nonzero `L_prior`, `L_gen`, and `L_boundary_extra` terms in
`total_loss`. It must not fall back to the old frozen-ref proxy.

This is not a final method result: Stage2 is not wired, full prior cache/full
training are not launched, and DAVIS/YouTubeVOS metrics do not exist yet.
