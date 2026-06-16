# VideoPainter Weight Resolution Report

Date: 2026-06-16

## Status

```text
status = resolved
```

The previous PAI blocker was missing VideoPainter / CogVideoX weights:

```text
third_party/VideoPainter/ckpt/CogVideoX-5b-I2V
third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch
```

PAI could not download from Hugging Face because its network was blocked, so
the weights were downloaded on HAL and transferred to PAI.

## HAL Download

Download directory:

```text
/home/hj/dpo-2-1-exp/third_party_baselines/VideoPainter/ckpt
```

Downloaded sources:

- `TencentARC/VideoPainter`
- `THUDM/CogVideoX-5b-I2V`

The fallback `zai-org/CogVideoX-5b-I2V` source was not needed.

Validated HAL layout:

```text
ckpt/CogVideoX-5b-I2V/model_index.json
ckpt/CogVideoX-5b-I2V/transformer/
ckpt/CogVideoX-5b-I2V/vae/
ckpt/CogVideoX-5b-I2V/tokenizer/
ckpt/CogVideoX-5b-I2V/text_encoder/
ckpt/VideoPainter/checkpoints/branch/config.json
ckpt/VideoPainter/checkpoints/branch/diffusion_pytorch_model.safetensors
```

HAL size check:

```text
VideoPainter: 1.3G
CogVideoX-5b-I2V: 21G
```

LFS pointer check:

```text
status = real_files_not_lfs_pointers
```

Detailed HAL logs:

```text
/home/hj/dpo-2-1-exp/third_party_baselines/VideoPainter/logs/hal_weight_validation_report.md
/home/hj/dpo-2-1-exp/third_party_baselines/VideoPainter/logs/hf_download_videopainter.log
/home/hj/dpo-2-1-exp/third_party_baselines/VideoPainter/logs/hf_download_cogvideox.log
```

## Transfer To PAI

Transfer method:

```text
rsync -avh --partial --append-verify
```

PAI target:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter/ckpt
```

Transfer report:

```text
/home/hj/dpo-2-1-exp/third_party_baselines/VideoPainter/logs/hal_to_pai_weight_transfer_report.md
```

PAI validation report:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/reports/videopainter_weight_resolution_report.md
```

PAI validation passed for the required base model and branch checkpoint paths.

## Result

The weight blocker is resolved. The Exp14 VideoPainter adapter trainer preflight
was rerun on PAI and passed, allowing the 2000-step gate to launch.
