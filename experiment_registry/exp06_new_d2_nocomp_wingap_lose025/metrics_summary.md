# Metrics Summary

Metric directory: `H20 qual_sbs_30 and reports/new_exp6_prompt_length_audit.md`

Metric policy:

- Video generation / full-mask generation: VBench where applicable.
- Video inpainting / partial-mask inpainting: project inpainting metrics through `inference/metrics.py` wrappers.

Current metric conclusion: New Exp6 is no-comp plus anchored loss, not plain Exp6; long training still needs caution.
