# Exp17 Saturation-Aware Positive DPO

Exp17 is a new isolated mainline gate after Exp11 outer b0.75 S2 and Exp16
Stage1-500. It does not continue OR, adapter work, adaptive normalization, or
Exp16 prior-confidence full training.

Goal:

```text
reduce DPO saturation / loser-dominant behavior while preserving winner quality
in mask and boundary regions
```

Base setting:

```text
Exp11 outer b0.75 S2
region-local normalized-gap clipped-loser-gap winner-anchored DPO
```

Variants:

- Exp17a: DPOP-style positive region loss.
- Exp17b: saturation-aware margin DPO.
- Exp17c: combined positive + saturation.

First gate:

```text
Stage1 1000 for each variant, then DAVIS10 visual / metric sanity.
No Stage2 is launched by this gate.
```
