# Exp15 OR Weight Download And Transfer Report

No new large weight download was required for the DAVIS50-only gate.

Verified existing PAI/NAS assets:

| Method | Weight path | Status |
|---|---|---|
| ProPainter | `/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter` | existing |
| DiffuEraser SFT-48000 | `/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000` | existing |
| Ours Exp11 outer b0.75 S2 | `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s2_2000_davis_pai/last_weights` | existing |
| COCOCO | `/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/COCOCO_weight` | existing |
| MiniMax-Remover | `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/minimax_remover/current` | existing, but env blocked |

HAL is not used as long-term storage in this DAVIS50 run. If future method
weights are downloaded through HAL, they should be transferred to PAI/NAS and
then removed from HAL temporary storage.

