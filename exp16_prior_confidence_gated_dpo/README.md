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

This folder intentionally blocks training until a manifest with real ProPainter
prior paths is available and `z_hat_x0` latent consistency is integrated into
the training loss. It must not fall back to the old frozen-ref proxy.

