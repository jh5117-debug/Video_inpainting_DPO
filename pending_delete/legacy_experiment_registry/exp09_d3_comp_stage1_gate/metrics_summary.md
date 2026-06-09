# Metrics Summary

Metric directory: `logs/target_eval/exp9_d3_comp_gate_pai_20260604_044925/metrics`

Metric policy:

- Video generation / full-mask generation: VBench where applicable.
- Video inpainting / partial-mask inpainting: project inpainting metrics through `inference/metrics.py` wrappers.

Current metric conclusion: Best metric checkpoint is ckpt500; longer steps degraded several metrics.
