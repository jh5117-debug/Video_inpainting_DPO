# Exp19b DAVIS10 Visual Case Judgement

Reviewed contact sheets for all ten DAVIS10 videos:

```text
boat, rhino, dog-agility, blackswan, lucia,
dance-jump, flamingo, soccerball, camel, car-roundabout
```

## Summary

Exp19b is visually safe but not visibly better than Exp11 outer b0.75 S2.
Across the ten videos, I did not see a reliable reduction in temporal flicker,
moving-boundary instability, mask texture artifacts, or outer-boundary seam
issues. I also did not see obvious new flow ghosting, double edges, or geometric
deformation.

## Classification

Better than Exp11:

```text
none
```

Tie / visually indistinguishable:

```text
boat
rhino
dog-agility
blackswan
lucia
soccerball
car-roundabout
```

Slight negative / no visible benefit with small metric regression:

```text
dance-jump
flamingo
camel
```

## Notes

- `dog-agility` has a tiny metric-positive signal, but the fast-motion boundary
  looks essentially the same as Exp11.
- `camel` is the clearest tiny negative case numerically, with no compensating
  visual improvement.
- The contact sheets support the metric conclusion: Exp19b is a neutral/negative
  gate and should not be expanded.
