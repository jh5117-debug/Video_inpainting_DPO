# PAI Post-Maintenance Permission Recheck - 2026-06-25

Host: `dsw-753014-85f54df947-bkp7h`

User: `hj`

## GPU State

All 8 L20X GPUs report `0 MiB` used and `0%` utilization. No compute processes
were reported during the recheck.

## Permission State

| Path | State |
| --- | --- |
| `/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000` | exists, but `hj` has no read/execute/write access |
| `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_vor_or_preference_data` | readable/executable/writable |
| `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2` | readable/executable/writable |
| `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp27_paper_grounded_preference_study` | readable/executable/writable |
| `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp25_vor_or_preference_data` | readable/executable, not writable |
| `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp26_videopainter_dpo_v2` | readable/executable, not writable |
| `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp27_paper_grounded_preference_study` | missing |
| `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime` | readable/executable/writable |

## Consequence

Exp26 Gate64 data preparation is complete, but DPO micro-training is still
blocked because the Exp26 experiment output root is not writable by `hj`.

Exp25 and Exp27 true DiffuEraser work remains blocked because the converted
DiffuEraser weights root is not readable by `hj`.
