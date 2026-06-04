# Metrics Summary

Metric directory: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/partialmask_eval/exp7_pm_dpoS1_sftS2_hybrid_20260602_025336/metrics`

Metric policy:

- Video generation / full-mask generation: VBench where applicable.
- Video inpainting / partial-mask inpainting: project inpainting metrics through `inference/metrics.py` wrappers.

Current metric conclusion: Hybrid with official/base Stage2 did not beat base; Stage2 handling remains risky.
