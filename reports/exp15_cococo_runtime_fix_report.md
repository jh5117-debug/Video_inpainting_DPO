# Exp15 CoCoCo Runtime Fix Report

Current status: `BLOCKED_NO_WEIGHT` / incomplete dependency.

## PAI Evidence

Checked paths:

```text
/mnt/nas/hj/official_repos/COCOCO_9ebe984                         exists
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/COCOCO_weight exists
```

Required SD inpainting dependency under:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/COCOCO_weight/stable-diffusion-v1-5-inpainting
```

Status:

| Required file/dir | Status |
|---|---|
| `model_index.json` | OK |
| `vae/config.json` | MISSING |
| `unet/config.json` | MISSING |
| `tokenizer/` | MISSING |
| `text_encoder/config.json` | MISSING |
| `scheduler/scheduler_config.json` | MISSING |

No complete `stable-diffusion-v1-5-inpainting` directory was found in the restricted PAI/NAS weight search.

## Conclusion

CoCoCo is not fixed in this pass. It remains `BLOCKED_NO_WEIGHT` until a complete SD inpainting diffusers directory is provided or downloaded.

## Safe Next Action

1. Download `runwayml/stable-diffusion-inpainting` or the exact CoCoCo-required SD inpainting dependency to HAL as transit.
2. Rsync to the CoCoCo weight directory on PAI/NAS.
3. Prefer an isolated CoCoCo env (`/mnt/nas/hj/conda_envs/cococo_or`) before running DAVIS50 OR.
