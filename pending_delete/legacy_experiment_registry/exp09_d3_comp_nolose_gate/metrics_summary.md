# Metrics Summary

Metric directory: `pending target eval`

Metric policy:

- Video generation / full-mask generation: VBench where applicable.
- Video inpainting / partial-mask inpainting: project inpainting metrics through `inference/metrics.py` wrappers.

Current metric conclusion: Training gate exists; diagnostic purpose is to test whether removing loser-gap reduces shortcut behavior.
