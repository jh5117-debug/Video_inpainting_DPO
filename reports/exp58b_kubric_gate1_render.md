# Exp58B Kubric Gate1 Render Smoke

Status: `VOID_NATIVE_KUBRIC_GATE1_READY`

## Scope

Gate1 generated exactly one official VOID Kubric paired sample. This milestone did not run VOID inference, preference forward, zero-gap, one-step, 10-step, or any training.

## Runtime

- Machine: PAI `dsw-753014-85f54df947-bkp7h`
- Blender: `/home/hj/tools/void_kubric_exp58b/blender-3.6.23-linux-x64/blender`
- Python env: `/home/hj/conda_envs/void_kubric_exp58b`
- Output root: `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/void/kubric_exp58b/gate1`
- Sample path: `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/void/kubric_exp58b/gate1/gate1/00000`
- Log: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp58_void_native_data_diagnostic/exp58b_kubric_env/gate1_render_retry_openexr32_staticffmpeg.log`

Command class:

```bash
blender -b --python blender_void_kubric_launcher.py -- \
  --job-dir .../kubric_exp58b/gate1 \
  --scratch_dir .../exp58b_kubric_env/scratch_gate1 \
  --kubasic_assets .../manifests/KuBasic.json \
  --gso_assets .../manifests/GSO.json \
  --hdri_assets .../manifests/HDRI_haven.json \
  --out_prefix gate1 \
  --num_pairs 1 \
  --start_index 0 \
  --resolution 128 \
  --fast \
  --seed 581
```

## Controlled Compatibility Fixes

Gate1 exposed two renderer-environment blockers after Blender/`bpy` was fixed:

1. `OpenEXR==3.4.13` segfaulted inside Blender Python while Kubric postprocessed rendered EXR layers.
2. PAI system `/usr/bin/ffmpeg` failed at encode time because it lacked `libblas.so.3`.

Controlled fixes:

- Downgraded only the isolated Exp58B env to `OpenEXR==3.2.10`.
- Added an Exp58B launcher-only OpenEXR channel-name shim for Blender's uppercase cryptomatte channels.
- Prepended `/home/hj/tools/void_kubric_exp58b/ffmpeg_bin` to `PATH`, where `ffmpeg` is a symlink to the `imageio-ffmpeg` static binary.

No base/system environment and no VOID official source files were modified.

## Outputs

Generated files:

- `rgb_full.mp4`
- `rgb_removed_objects_invisible.mp4`
- `rgb_altered_physics.mp4`
- `mask.mp4`
- `metadata.json`

The official script names the physically removed winner as `rgb_altered_physics.mp4`; this is the file to normalize as `rgb_removed` in later manifests if needed.

## Validation

All videos decode:

- Frames: 24
- Resolution: 128x128
- FPS: 8
- `rgb_full.mp4` SHA256: `281de3ab8e1dc8aa4943b4a2e23b98575be6f3de3b646296836e34fd48956011`
- `rgb_removed_objects_invisible.mp4` SHA256: `ed2525ef9da56af286794b2e421ffcc73e8cce29c503aaa6a27074c6914cae90`
- `rgb_altered_physics.mp4` SHA256: `1e6300a459358ef1a26b2fd8e9a72ef24ea36f9e778be458af4530d01d66f131`
- `mask.mp4` SHA256: `1ee8597e18e8b3ae8764314755243de8834f11686fb2990bbeb344022fb587e6`

Aggregate mask counts across all frames:

- value 0: 1312 pixels, 0.3337%
- value 63: 420 pixels, 0.1068%
- value 127: 11797 pixels, 3.0001%
- value 255: 379687 pixels, 96.5594%

Metadata:

- `num_objects`: 3
- `num_removed`: 1
- `target_hit`: false
- `background`: `abandoned_games_room_02`
- `camera_motion`: `zoom_in`

Gate1 is a renderer smoke pass. The low target-hit value means it is not a data-quality conclusion by itself.

## Visual Review

Opened and reviewed:

- `side_by_side.jpg`
- `temporal_strip_16f.jpg`
- `object_crop_sheet.jpg`
- `affected_overlap_crop_sheet.jpg`
- `boundary_crop_sheet.jpg`
- `outside_crop_sheet.jpg`
- `temporal_diff_heatmap.jpg`
- `mask_histogram_panel.jpg`

Visual result: valid low-resolution Kubric object-removal / altered-physics pair. Object, affected, overlap, and background regions are non-empty in the aggregate mask. No fake or VOR-derived data was mixed in.

## Decision

Gate1 passes as `VOID_NATIVE_KUBRIC_GATE1_READY`. Gate8 generation is unlocked, but only exactly N=8 may run next. No VOID inference or training is allowed in Exp58B.
