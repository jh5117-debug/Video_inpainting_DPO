# Exp38 Metric Summary

No Exp38 metrics have been generated yet.

Readback imported prior MiniMax metric status:

- Exp30 frozen/EMA 10-step: near-tie/slightly negative heldout metrics and
  `0/32` visual better rows.
- Exp35 hard-noise rescue: all three recipes had negative mask/boundary/outside
  mean deltas and `0/48` visual better rows.
- Exp36 winner-SFT: training loss decreased and outputs moved, but heldout
  quality was not positive.
- Exp37 R1 had mixed numeric movement
  (`+0.200826` full PSNR, `+0.161946` mask PSNR, `-0.049755` boundary PSNR)
  and only `1/16` visually better heldout rows.

Exp38 metrics remain pending future preregistered milestones.

## 2026-06-28 Failure Taxonomy

No new metrics were generated. The taxonomy interprets prior metrics:

- Exp30/35 failures were near-tie or negative despite strict reload.
- Exp36 sensitivity showed nonzero output response to weight perturbation.
- Exp36 winner-SFT lowered train loss but did not improve heldout metrics.
- Exp37 R1 was the only mixed-positive numeric signal but remained below the
  visual gate.

## 2026-06-28 Train-Overfit Diagnosis

New Exp38 inference metrics were generated on PAI GPU0/GPU1.

Exp37 R1 LocalDPO-badnoise checkpoint-10:

- Train32 full/mask/boundary/outside PSNR deltas:
  `-0.586255` / `+0.152062` / `+0.069123` / `-0.895018`.
- Heldout16 full/mask/boundary/outside PSNR deltas:
  `+0.200826` / `+0.161946` / `-0.049755` / `+0.028198`.

Exp36 S1 winner-SFT checkpoint-10:

- Train32 full/mask/boundary/outside PSNR deltas:
  `+0.016242` / `-0.006448` / `-0.001681` / `+0.024897`.
- Heldout16 full/mask/boundary/outside PSNR deltas:
  `-0.010218` / `-0.008293` / `-0.010939` / `-0.014499`.

Conclusion: R1 confirms output movement but with local/global tradeoffs; S1 is
too weak to count as train-overfit. No quality-positive or 30-step gate is
unlocked.
