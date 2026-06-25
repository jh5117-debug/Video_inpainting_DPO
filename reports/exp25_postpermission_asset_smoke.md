# Exp25 Post-Permission DiffuEraser Asset Smoke

Date: 2026-06-25

Status: `POSTPERMISSION_ASSET_SMOKE_PASSED`

This is a technical asset/wrapper gate only. It does not establish loser utility, `DATA_READY`, or scientific positivity.

## Permission and Inventory

The fixed root-cause sample manifest was read successfully:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_vor_or_preference_data/root_cause_matrix_20260625/root_cause_sample12_manifest.jsonl`

Manifest SHA256:

`d1a7ef848ce1f5777ae80f1655c581fa5328d108fab497693d8afddf750afa49`

The no-forward asset smoke verified readable/traversable access to:

- `/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000`
- `/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000/brushnet/config.json`
- `/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000/unet_main/config.json`
- `/mnt/workspace/hj/nas_hj/weights/stable-diffusion-v1-5`
- `/mnt/workspace/hj/nas_hj/weights/sd-vae-ft-mse`
- `/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter`

## Real Inference Smoke

Sample:

`REAL_ENV114_00004_004_02`

Generator:

`diffueraser_or_none_propainter_2f2ae30313f3`

Configuration:

- `pcm_mode=none`
- `prior_mode=propainter`
- `no_pcm_steps=6`
- `no_pcm_guidance=0.0`
- `mask_dilation_iter=0`
- `raw_no_comp=true`
- `hard_comp=false`
- `seed=20260625`
- `width=512`
- `height=288`
- `num_frames=24`

Result:

- return code: `0`
- frames written: `24`
- failed rows: `0`
- fallback: `false`
- elapsed: `238.5769s`

Output:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_vor_or_preference_data/postpermission_asset_smoke_20260625_130502/diffueraser/raw_frames/REAL_ENV114_00004_004_02`

Log:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_vor_or_preference_data/postpermission_asset_smoke_20260625_130502/diffueraser/logs/REAL_ENV114_00004_004_02.log`

## Decision

The previous DiffuEraser checkpoint permission blocker is resolved. Exp25 may proceed to the fixed 12-sample root-cause matrix. Gate128 and OR-DPO remain not started.

