# Exp29 EffectErase Official 81F Command Validation

Status: `EFFECTERASE_OFFICIAL81_COMMAND_READY`

- Manifest: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_official81_source_audit_20260627/manifests/effecterase_smoke_official81_preregistered.jsonl`
- Manifest SHA256: `706cb09286fd8528d7efbbb91eb89673a9ec7ce61b0047e6b3b2e8ea4c9b1fb3`
- Rows validated: 8
- Repo: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/third_party/EffectErase`
- Python: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/venvs/effecterase_smoke/bin/python`
- Asset root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_weight_recovery_20260626`
- Assets ready: True
- Inputs ready: True
- Official help return code: 0
- Supports `--num_frames`: True
- VOR-Eval use: False
- Training eligibility: False
- Dry-run only: True

## Constructed Command Example

```bash
cd /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/third_party/EffectErase && CUDA_VISIBLE_DEVICES=<RIGHT_GPU> /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/venvs/effecterase_smoke/bin/python examples/remove_wan/infer_remove_wan.py --fg_bg_path /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_official81_20260626/inputs/REAL_ENV005_00003_003_05/condition_81f.mp4 --mask_path /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_official81_20260626/inputs/REAL_ENV005_00003_003_05/mask_81f.mp4 --output_path /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_official81_20260626/outputs/REAL_ENV005_00003_003_05/raw_output.mp4 --text_encoder_path /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_weight_recovery_20260626/Wan-AI/Wan2.1-Fun-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth --vae_path /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_weight_recovery_20260626/Wan-AI/Wan2.1-Fun-1.3B-InP/Wan2.1_VAE.pth --dit_path /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_weight_recovery_20260626/Wan-AI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors --image_encoder_path /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_weight_recovery_20260626/Wan-AI/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth --pretrained_lora_path /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_weight_recovery_20260626/FudanCVL/EffectErase/EffectErase.ckpt --num_frames 81 --height 480 --width 832 --seed 2025 --cfg 1.0 --num_inference_steps 50
```

No full EffectErase inference was launched by command validation.
