# official_videodpo_diffueraser_youtubevos_partialmask_data

Purpose: data-source ablation that extends the partial-mask preference-data construction to YouTube-VOS.

## Definition

Changed:

- Data source changes from original VideoDPO preference pairs to YouTube-VOS-derived video/mask data.
- Partial-mask inpainting preference data should follow the comp setup from `official_videodpo_diffueraser_data_partialmask_loser_comp`.
- Training is based on the `official_videodpo_diffueraser_task_partialmask` setting, so DiffuEraser receives partial masks during training.

Not changed:

- First implementation should reuse the same manifest schema.
- DPO loss and metrics are not changed.
- First version should use comp. No-comp is a later diagnostic.

## Current Status

Scaffold only. The audit found local `data/external/youtubevos_432_240`, but PAI YouTube-VOS path was not mounted/confirmed in this session.

## Data Contract

| Field | Value |
| --- | --- |
| data source | YouTube-VOS |
| win | clean / target YouTube-VOS clip |
| loser | partial-mask inpainting generated loser |
| comp | true in first version |
| mask for loser generation | partial |
| mask for training | partial |
| changed variable | data source |

Clip sampling should reuse or audit existing YouTube-VOS SFT / DiffuEraser data code before implementation.

## Metrics / Diagnostics

Use the same PSNR, SSIM, VBench, SBS, and DPO diagnostics policy as the VideoDPO-source partial-mask experiments.
