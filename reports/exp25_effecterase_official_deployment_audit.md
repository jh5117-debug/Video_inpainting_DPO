# Exp25 EffectErase Official Deployment Audit

Date: 2026-06-23

## Official Repo

Repo:

`https://github.com/FudanCVL/EffectErase.git`

Local clone:

`/home/hj/video_inpainting_third_party/EffectErase`

Commit:

`bcee0a5da5ef387c2ba39390dc4d579503669fb8`

License:

Creative Commons Attribution-NonCommercial 4.0 International.

## Official Command Path

Official README directs:

- `pip install -e .`
- `hf download alibaba-pai/Wan2.1-Fun-1.3B-InP --local-dir Wan-AI/Wan2.1-Fun-1.3B-InP`
- `hf download FudanCVL/EffectErase EffectErase.ckpt --local-dir ./`
- `bash script/test_remove.sh`

Official `script/test_remove.sh` launches:

`examples/remove_wan/infer_remove_wan.py`

Required assets from `Wan2.1-Fun-1.3B-InP`:

- `models_t5_umt5-xxl-enc-bf16.pth`
- `Wan2.1_VAE.pth`
- `diffusion_pytorch_model.safetensors`
- `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`

Required EffectErase checkpoint:

- `EffectErase.ckpt`

## Asset Scan

Scanned roots:

- `/home/hj`
- `/mnt/nas/hj`
- `/mnt/workspace/hj`

Result:

- `EffectErase.ckpt`: not found.
- `Wan2.1-Fun-1.3B-InP`: not found.
- official EffectErase repo: found on HAL only.

## Current Gate Status

`EFFECTERASE_OFFICIAL_ASSETS_MISSING`

No EffectErase VOR smoke can be marked valid yet. The previous Exp25 smoke did not run EffectErase and no fallback wrapper was used.

## Next Valid Steps

1. Download official assets on HAL if Hugging Face access allows it.
2. Record size and SHA256 for each asset.
3. Rsync assets and the clean official repo to PAI.
4. Build `/mnt/nas/hj/conda_envs/effecterase` with `pip install -e .`.
5. Run EE-L0 official example exactly.
6. Only after EE-L0 passes, run EE-L1 VOR one-sample and EE-L2 six-sample smoke.

EffectErase remains blocked for Gate128 until EE-L0 through EE-L2 are complete and visually reviewed.
