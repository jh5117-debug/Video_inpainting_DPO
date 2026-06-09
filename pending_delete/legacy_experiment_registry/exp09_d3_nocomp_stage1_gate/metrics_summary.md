# Metrics Summary

Metric directory: `/home/nvme01/H20_Video_inpainting_DPO/logs/target_eval/exp9_d3_nocomp_gate_h20_20260604_023243/metrics`

Metric policy:

- Video generation / full-mask generation: VBench where applicable.
- Video inpainting / partial-mask inpainting: project inpainting metrics through `inference/metrics.py` wrappers.

Current metric conclusion: Completed and evaluable; qualitative review suggests both base and DPO can look poor on target-domain eval, so metrics need qualitative caveat.
