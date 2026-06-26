# Exp29 EffectErase Command Dry-Run

Date: 2026-06-26

Status: `EFFECTERASE_COMMAND_READY`

## Summary

- Runtime repo path: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/third_party/EffectErase`
- Expected repo commit: `bcee0a5da5ef387c2ba39390dc4d579503669fb8`
- License: `CC BY-NC 4.0`
- Venv Python: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/venvs/effecterase_smoke/bin/python`
- Transformers version: `4.51.3`
- Diffusers version: `0.31.0`
- Official script: `examples/remove_wan/infer_remove_wan.py`
- Supports `--num_frames`: `True`
- Supports `--cfg`: `True`
- Supports `--num_inference_steps`: `True`
- Supports `--seed`: `True`

## Asset Identity

| asset | bytes | sha256 | path |
| --- | ---: | --- | --- |
| lora | 1438679011 | `4e9ace4607a348a0d820328f827f7f4f751a0e44d4cf60665c03e9740dfa024d` | `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_weight_recovery_20260626/FudanCVL/EffectErase/EffectErase.ckpt` |
| vae | 507609880 | `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981` | `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_weight_recovery_20260626/Wan-AI/Wan2.1-Fun-1.3B-InP/Wan2.1_VAE.pth` |
| dit | 3128957992 | `8495d2b1673ffb18abb548a64ff3b0e4bd367734f653096f7a8a3ad46954d511` | `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_weight_recovery_20260626/Wan-AI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors` |
| image_encoder | 4772359047 | `628c9998b613391f193eb67ff68da9667d75f492911e4eb3decf23460a158c38` | `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_weight_recovery_20260626/Wan-AI/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` |
| text_encoder | 11361920418 | `7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d` | `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_weight_recovery_20260626/Wan-AI/Wan2.1-Fun-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth` |

## Constructed Command

The command below is constructed from the locked preregistered protocol and a ready row. It was not executed as full inference in this milestone.

```bash
cd /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/third_party/EffectErase && CUDA_VISIBLE_DEVICES=<RIGHT_GPU> /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/venvs/effecterase_smoke/bin/python examples/remove_wan/infer_remove_wan.py --fg_bg_path /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_20260626/inputs/REAL_ENV231_00010_003_03/fg_bg.mp4 --mask_path /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_20260626/inputs/REAL_ENV231_00010_003_03/mask.mp4 --output_path /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_20260626/outputs/REAL_ENV231_00010_003_03/raw_output.mp4 --text_encoder_path /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_weight_recovery_20260626/Wan-AI/Wan2.1-Fun-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth --vae_path /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_weight_recovery_20260626/Wan-AI/Wan2.1-Fun-1.3B-InP/Wan2.1_VAE.pth --dit_path /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_weight_recovery_20260626/Wan-AI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors --image_encoder_path /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_weight_recovery_20260626/Wan-AI/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth --pretrained_lora_path /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_weight_recovery_20260626/FudanCVL/EffectErase/EffectErase.ckpt --num_frames 17 --height 480 --width 832 --seed 2025 --cfg 1.0 --num_inference_steps 50
```

## Decision

- Input status: `EFFECTERASE_SMOKE_INPUTS_BLOCKED` (5/6 ready)
- Inference allowed by inputs: `False`
- Full inference run: `False`
- Reason: Milestone A input materialization is blocked for the locked 6-row smoke; command is ready but C cannot run as preregistered 6/6 smoke.

Command/environment validation is ready, but the official six-row smoke remains blocked until a new preregistration or corrected non-empty-mask input set is explicitly authorized.
