# Exp31 VideoPainter Step0/50/2000 Evaluation Summary

Status: `VIDEOPAINTER_2000_STEP0_50_2000_EVALUATED`

- run_root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp31_videopainter_2000step_longrun/exp31_vp2000_eval_step0_50_2000_20260628_032700`
- csv: `/home/hj/runtime_code/H20_Video_inpainting_DPO_exp31_vp2000/reports/exp31_vp_2000_step0_50_2000_eval_summary.csv`

| split | step | rows | ok | full_psnr | mask_psnr | medium-hard | hard-plausible | trivial-bad | invalid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| search | 0 | 32 | 32 | 22.6866 | 16.1617 | 27 | 4 | 1 | 0 |
| search | 50 | 32 | 32 | 22.1228 | 24.2617 | 32 | 0 | 0 | 0 |
| search | 2000 | 32 | 32 | 28.2567 | 26.1364 | 32 | 0 | 0 | 0 |
| shadow | 0 | 32 | 32 | 21.5685 | 15.2466 | 26 | 5 | 1 | 0 |
| shadow | 50 | 32 | 32 | 21.3544 | 24.0494 | 32 | 0 | 0 | 0 |
| shadow | 2000 | 32 | 32 | 27.8317 | 26.1325 | 31 | 0 | 0 | 0 |

This report evaluates completed checkpoints only. It does not start new VideoPainter training or claim final SOTA.
