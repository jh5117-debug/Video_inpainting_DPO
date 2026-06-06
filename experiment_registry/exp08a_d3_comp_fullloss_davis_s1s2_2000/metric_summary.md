# Metric Summary

Metric path:

Stage1: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp08a_fullloss_stage1_val_davis_20260606_070556/metrics`

Stage2: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp08a_fullloss_stage2_val_davis_20260606_070556/metrics`

Status: complete and negative.

| validation | model | boundary_psnr | boundary_ssim | mask_psnr | mask_ssim | whole_psnr | whole_ssim |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage1 | DPO-S1_SFT-S2 | 16.1306 | 0.4964 | 15.6757 | 0.4813 | 23.9554 | 0.9017 |
| Stage1 | DiffuEraser-base | 23.1742 | 0.7861 | 22.7633 | 0.7754 | 29.4647 | 0.9564 |
| Stage2 | DPO-S1_DPO-S2 | 15.7133 | 0.4790 | 15.2577 | 0.4638 | 23.5677 | 0.8967 |
| Stage2 | DiffuEraser-base | 23.0682 | 0.7804 | 22.6570 | 0.7695 | 29.3802 | 0.9558 |

Conclusion: both DPO variants underperform the SFT-48000 DiffuEraser baseline on DAVIS. Stage2 DPO is slightly worse than Stage1 validation.

Metric policy: partial-mask inpainting metrics must use `tools/run_inpainting_metric_eval.py` / `inference/metrics.py`. Do not use VBench.
