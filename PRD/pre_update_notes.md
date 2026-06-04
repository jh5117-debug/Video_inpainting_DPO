# Pre Update Notes

Updated: 2026-06-04

## Required Story Corrections

- `converted_weights_step48000` is the fine-tuned YouTube-VOS SFT-48000 DiffuEraser baseline, not an ordinary naked base.
- Exp4 is a failed fullmask generated-loser quality gate, not a successful training result.
- Old Exp5, Exp5 beta10 plain, New Exp5, and New Exp6 must be separated.
- New Exp6 is D2 no-comp + winner-gap regularized DPO. There was no plain Exp6 first.
- Exp7 is suspicious and must be audited for ProPainter prior, mask area, and eval path before further D3 expansion.
- DPO diagnostics are mandatory evidence for every DPO experiment.

## Exp7 Fix Policy

Before more target-domain DPO:

1. Generate small-mask VideoDPO data at 15%-20% area with ProPainter prior.
2. Run Stage1-only gates only.
3. Do not run DPO Stage2.
4. Do not use VBench for partial-mask inpainting.
5. Use four-column qualitative videos.

## Target-Domain Principle

Training may use YouTube-VOS/D3, but final target eval should use DAVIS:

`/mnt/workspace/hj/nas_hj/data/external/davis_432_240`
