# Exp58B Kubric Gate8 Generation

Status: `VOID_NATIVE_KUBRIC_GATE8_READY`

## Scope

Generated exactly 8 official Kubric native paired samples. No Gate16/Gate32 expansion was run. This milestone did not run VOID inference, preference forward, zero-gap, one-step, 10-step, or training.

## Runtime

- Machine: PAI `dsw-753014-85f54df947-bkp7h`
- Blender: `/home/hj/tools/void_kubric_exp58b/blender-3.6.23-linux-x64/blender`
- Env: `/home/hj/conda_envs/void_kubric_exp58b`
- Output root: `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/void/kubric_exp58b/gate8`
- Sample root: `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/void/kubric_exp58b/gate8/gate8`
- Review root: `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/void/kubric_exp58b/gate8/review_pages`
- Log: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp58_void_native_data_diagnostic/exp58b_kubric_env/gate8_render.log`
- Manifest: `manifests/exp58b_void_native_kubric_gate8.jsonl`

## Generation Config

- `--num_pairs 8`
- `--start_index 0`
- `--resolution 128`
- `--fast`
- `--out_prefix gate8`
- local official manifests for KuBasic, GSO, and HDRI Haven
- isolated OpenEXR channel shim enabled
- static `imageio-ffmpeg` first on PATH

## Render Result

- Successful: 8/8
- Failed: 0/8
- Frame count: 24 for every sample
- Resolution: 128x128 for every sample
- Decode status: pass for every sample
- Quadmask values: every sample contains 0/63/127/255 in aggregate
- VOR data mixed: no
- Fake data created: no

## Region Ranges

- Object area: 127 to 7331 pixels
- Overlap area: 46 to 6095 pixels
- Affected area: 1715 to 40515 pixels
- Background area: 345378 to 391078 pixels

## Caveat

All 8 `metadata.json` files report `target_hit=false`. The generated videos still contain non-empty affected and overlap masks from the C-vs-B RGB difference path, but Gate8 should be treated as a native-data generation diagnostic, not as proof of high-quality collision/interaction training data.

## Visual Review

Opened all 8 review pages:

- `00000_review_page.jpg`
- `00001_review_page.jpg`
- `00002_review_page.jpg`
- `00003_review_page.jpg`
- `00004_review_page.jpg`
- `00005_review_page.jpg`
- `00006_review_page.jpg`
- `00007_review_page.jpg`

Each page includes temporal full/winner strips, full/invisible/winner/mask mid-frame panels, object crop, affected/overlap crop, boundary crop, outside crop, and mask histogram.

## Decision

Gate8 is ready as `VOID_NATIVE_KUBRIC_GATE8_READY`. The next experiment may run official VOID inference on this native Kubric Gate8. Exp58B itself stops here: no inference, no one-step, and no 10-step.
