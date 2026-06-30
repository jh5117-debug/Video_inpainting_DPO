# Exp50 VOID Weight Download

Status: `VOID_WEIGHT_DOWNLOAD_BLOCKED`.

## Attempts

| Repo | Target | Attempt | Status | Log |
| --- | --- | --- | --- | --- |
| `netflix/void-model` | `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/void/netflix_void-model` | `hf download netflix/void-model void_pass1.safetensors void_pass2.safetensors` | `NETWORK_BLOCKED` | `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp50_pai_void_adapter_feasibility/b2_void_official_hf_download_20260630_104210.log` |
| `alibaba-pai/CogVideoX-Fun-V1.5-5b-InP` | `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/void/CogVideoX-Fun-V1.5-5b-InP` | `hf download --dry-run alibaba-pai/CogVideoX-Fun-V1.5-5b-InP` | `NETWORK_BLOCKED` | `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp50_pai_void_adapter_feasibility/b2_base_official_hf_probe_20260630_104444.log` |

Both attempts used the official HuggingFace endpoint (`HF_ENDPOINT` unset). No mirror, fallback, copied asset, or fabricated file was used.

## Error

Both attempts failed with official HF network access errors:

```text
httpx.ConnectError: [Errno 101] Network is unreachable
```

The base-model dry-run also raised:

```text
huggingface_hub.errors.DryRunError: Dry run cannot be performed as the repository cannot be accessed.
```

## Result

No Pass1, Pass2, or CogVideoX-Fun base model weights were downloaded. Milestone C/D/E/F remain blocked until the exact assets are available.

## Required Missing Assets

- `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/void/netflix_void-model/void_pass1.safetensors`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/void/netflix_void-model/void_pass2.safetensors`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/void/CogVideoX-Fun-V1.5-5b-InP`
