# PAI Asset Smoke Report

- generated_at: 2026-05-24T01:27:53+08:00
- repo_root: /mnt/nas/hj/H20_Video_inpainting_DPO
- out_root: /mnt/nas/hj/H20_Video_inpainting_DPO/outputs/asset_smoke_tests/20260524_012753
- run_one_sample: 0

| Model | Inference Script | Weight Root | Python Compile | Weight Listing | One-sample Generation |
| --- | --- | --- | --- | --- | --- |
| diffueraser | `DPO_finetune/infer_diffueraser_candidate.py` | `/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/diffueraser/Orign_Diffueraser` | OK | OK | NOT_RUN |
| propainter | `DPO_finetune/infer_propainter_candidate.py` | `/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter` | OK | OK | NOT_RUN |
| cococo | `DPO_finetune/infer_cococo_candidate.py` | `/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/COCOCO_weight` | OK | OK | NOT_RUN |
| minimax_remover | `DPO_finetune/infer_minimax_candidate.py` | `/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/minimax` | OK | OK | NOT_RUN |

## Notes

- Default mode does import/compile and weight-path smoke only.
- Set `RUN_ONE_SAMPLE=1` only after confirming the exact model-specific command.
- This script intentionally does not start DPO training.
