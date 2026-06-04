# Metrics Summary

Metric directory: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/partialmask_eval/exp7_gate1500_20260602_000500/metrics`

Metric policy:

- Video generation / full-mask generation: VBench where applicable.
- Video inpainting / partial-mask inpainting: project inpainting metrics through `inference/metrics.py` wrappers.

Current metric conclusion: Task alignment helps Stage1, but DPO Stage2 regresses.
