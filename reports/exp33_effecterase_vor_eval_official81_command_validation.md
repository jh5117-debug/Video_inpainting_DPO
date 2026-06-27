# Exp33 EffectErase VOR-Eval Official81 Command Validation

Status: `EXP33_VOREVAL_OFFICIAL81_COMMAND_READY`

- Branch: `research/exp33-effecterase-vor-eval-baseline-20260627`
- Commit: `1a8c3e44535b64f4c1f69ce8ae6a8b8d7aefccd1`
- Manifest: `/home/hj/runtime_code/H20_Video_inpainting_DPO_exp33_effecterase_eval/exp33_effecterase_vor_eval_baseline/manifests/effecterase_vor_eval_official81_ready.jsonl`
- Manifest SHA256: `d5dc6052aae897ff01dcc2af8209de51dfbd04caf3f37534f0940c1f11a94811`
- Rows validated: 43
- Expected rows: 43
- Repo: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/third_party/EffectErase`
- Python: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/venvs/effecterase_smoke/bin/python`
- Asset root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_weight_recovery_20260626`
- Output root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp33_effecterase_vor_eval_baseline/vor_eval_official81_compat_20260627_134945`
- Runtime dir: `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/idle_gpu_parallel_20260627/exp33_effecterase_vor_eval_official81`
- Assets ready: True
- Inputs ready: True
- Official help return code: 0
- Supports `--num_frames`: True
- VOR-Eval rows required: True
- Training eligible rows present: False
- Adapter/training launched: false
- Dry-run only: False

## Constructed Command Example

```bash
cd /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/third_party/EffectErase && CUDA_VISIBLE_DEVICES=<NON_RESERVED_GPU> /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/venvs/effecterase_smoke/bin/python examples/remove_wan/infer_remove_wan.py --fg_bg_path /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp33_effecterase_vor_eval_baseline/vor_eval_official81_compat_20260627_134945/inputs/REAL_ENV900_00001_001_03/condition_81f.mp4 --mask_path /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp33_effecterase_vor_eval_baseline/vor_eval_official81_compat_20260627_134945/inputs/REAL_ENV900_00001_001_03/mask_81f.mp4 --output_path /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp33_effecterase_vor_eval_baseline/vor_eval_official81_compat_20260627_134945/outputs/REAL_ENV900_00001_001_03/raw_output.mp4 --text_encoder_path /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_weight_recovery_20260626/Wan-AI/Wan2.1-Fun-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth --vae_path /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_weight_recovery_20260626/Wan-AI/Wan2.1-Fun-1.3B-InP/Wan2.1_VAE.pth --dit_path /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_weight_recovery_20260626/Wan-AI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors --image_encoder_path /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_weight_recovery_20260626/Wan-AI/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth --pretrained_lora_path /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_weight_recovery_20260626/FudanCVL/EffectErase/EffectErase.ckpt --num_frames 81 --height 480 --width 832 --seed 2025 --cfg 1.0 --num_inference_steps 50
```

This runner is restricted to held-out VOR-Eval EffectErase raw baseline inference.
It does not launch adapter training and refuses training-eligible rows.
