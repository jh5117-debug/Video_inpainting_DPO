# Video Inpainting DPO Experiment Registry

This registry is the source-of-truth artifact ledger for the project. Every experiment has an independent folder with config, command notes, status, paths, metrics, qualitative notes, and DPO diagnostics notes.

Key corrections:

- Old Exp5 and New Exp5 are different experiments.
- New Exp6 is no-comp + winner-anchored loss, not plain Exp6.
- Exp7 current is suspicious and must be fixed with small masks and ProPainter prior before expanding D3.
- YouTube-VOS/D3 work must use `converted_weights_step48000`, the YouTube-VOS SFT-48000 DiffuEraser weights, not a naked base.
- Partial-mask inpainting should use ProPainter prior; fullmask/video-generation settings should not rely on it.
- DPO diagnostics are mandatory evidence; missing diagnostics are explicitly marked `MISSING_DPO_DIAG`.
