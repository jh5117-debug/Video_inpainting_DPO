# Exp29 MiniMax Repo And Weight Audit

Date: 2026-06-26

## Status

- `MINIMAX_REPO_READY`
- `MINIMAX_WEIGHTS_READY`
- `MINIMAX_INFERENCE_SMOKE_PENDING`
- `MINIMAX_TRAINABLE_FORWARD_PENDING`

MiniMax is not an adapter success yet. It is ready for an isolated official
inference smoke on PAI because repo code and PAI/NAS weights are present.

## Repository

- Local repo: `/home/hj/dpo-2-1-exp/third_party_baselines/MiniMax-Remover/repo`
- Remote: `https://github.com/zibojia/MiniMax-Remover`
- Commit: `28e12b450d8a72a7547b86940a4985e6ad90d75b`
- Dirty status: clean in the audited repo
- License file: not present in the local repo root
- README model card: `https://huggingface.co/zibojia/minimax-remover`

## Code Evidence

Official/local code includes:

- `pipeline_minimax_remover.py`
- `transformer_minimax_remover.py`
- `test_minimax_remover.py`

The pipeline loads:

- `AutoencoderKLWan`
- `Transformer3DModel`
- `FlowMatchEulerDiscreteScheduler` or `UniPCMultistepScheduler`

The inference path concatenates current latents, masked video latents, and mask
latents before the transformer forward. The local repo is inference-oriented:
no full official training script or minimax training objective is present in the
repo root.

## Paper Evidence

Local paper file:

`/home/hj/Video Inpainting/Minimax-Remover.pdf`

The paper describes a flow-matching objective with:

- noisy latent interpolation: `z_t = t * epsilon + (1 - t) * z_0`
- target velocity: `v = epsilon - z_0`
- Stage 2 minimax training that searches adversarial "bad noise" and then
  trains the model to be robust to it.

This supports MiniMax as a plausible flow-style adapter candidate, but the
adapter implementation must still prove native target parity before any DPO
claim.

## Weights

HAL temporary weights were removed after a previous validation. The remaining
HAL note is:

`/home/hj/dpo-2-1-exp/third_party_baselines/MiniMax-Remover/weights/README_REMOVED_TEMP_WEIGHTS.txt`

PAI/NAS weights are present through:

`/mnt/nas/hj/H20_Video_inpainting_DPO/weights/minimax_remover/current`

The symlink target is:

`/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/minimax`

Required file identity:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `vae/config.json` | 724 | `f0c1cc1d7decb5badc384f54691746a27a9aeff49f7ebca974e583389342d527` |
| `vae/diffusion_pytorch_model.safetensors` | 507591892 | `d6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793` |
| `transformer/config.json` | 422 | `7d218c1d52a04e9e4f0e89ca72c9743daa408177e3bdccde90de1334801f8f77` |
| `transformer/diffusion_pytorch_model.safetensors` | 2254157576 | `a379d98432970f614befb260357153edcd01a99748cf7f6dabe1a230c159b213` |
| `scheduler/scheduler_config.json` | 751 | `3fed2abbd9bbc301a74db01947198057ec5049808910dccab320925bf27bea6e` |

## Adapter Feasibility

MiniMax can be audited as a true adapter candidate because it is a differentiable
flow-style transformer pipeline with a paper-defined velocity target. However,
Exp29 has not yet run:

- official inference smoke;
- one-row trainable forward;
- policy/reference zero-gap;
- one-step strict reload;
- 10-step micro gate.

No DPO or optimizer step has been launched in this milestone.

