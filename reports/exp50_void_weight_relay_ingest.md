# Exp50 VOID Weight Relay Ingest

Status: `VOID_WEIGHTS_READY`.

The H20/HAL relay artifacts were ingested on PAI without re-downloading weights. This step only verifies file presence and SHA256 equivalence; it does not load models, run inference, or use GPU.

## Source Roots

- VOID weights: `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/void/netflix_void-model`
- CogVideoX-Fun base weights: `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/void/CogVideoX-Fun-V1.5-5b-InP`
- Relay reports: `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/void/relay_reports`
- Relay SHA: `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/void/relay_sha256`

## Required Checks

- VOID files: 12 / 12
- Base files: 40 / 40
- `void_pass1.safetensors`: present
- `void_pass2.safetensors`: present
- Base transformer safetensors: present
- Base VAE safetensors: present
- Base text encoder shards: present
- Relay SHA exists: yes
- PAI SHA exists: yes
- Hash match: yes
- Missing files: 0
- Extra files: 0
- Hash mismatches: 0

## Decision

`VOID_WEIGHTS_READY`: PAI Exp50 can proceed to environment/import smoke if ready. The previous `VOID_WEIGHT_DOWNLOAD_BLOCKED` condition is resolved by official HF relay plus PAI-side SHA verification.
