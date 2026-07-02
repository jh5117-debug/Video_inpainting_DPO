# Exp58B Blender / bpy Smoke

Status: `EXP58B_BLENDER_BPY_READY`

## Scope

This milestone recovered only the isolated Blender/`bpy` renderer environment for official VOID Kubric generation. It did not run VOID inference, training, preference forward, zero-gap, one-step, or 10-step.

## Install Layout

- Machine: PAI `dsw-753014-85f54df947-bkp7h`
- Tools root: `/home/hj/tools/void_kubric_exp58b`
- Blender tarball: `/home/hj/tools/void_kubric_exp58b/blender_download/blender-3.6.23-linux-x64.tar.xz`
- Blender executable: `/home/hj/tools/void_kubric_exp58b/blender-3.6.23-linux-x64/blender`
- Python env: `/home/hj/conda_envs/void_kubric_exp58b`
- Kubric source: `/home/hj/tools/void_kubric_exp58b/kubric_src`

The preferred NAS tools/env roots were not writable by `hj`, so Exp58B uses isolated `/home/hj` fallback roots for tools and Python packages while keeping render outputs planned for NAS data/log/runtime roots.

## Controlled Shared Libraries

Blender 3.6.23 initially failed on PAI with missing `libSM.so.6` and `libICE.so.6`. No system packages were installed. The two Ubuntu Noble packages were downloaded with `apt-get download` and extracted under:

`/home/hj/tools/void_kubric_exp58b/user_libs/usr/lib/x86_64-linux-gnu`

Runtime uses `LD_LIBRARY_PATH` pointing at that user library directory.

## Hashes

- Blender tarball SHA256: `0e9a18af4d0060b825e9617e24a775f759e0f9f67271c062f3d53a539030af00`
- `libice6` deb SHA256: `ad1edb5303574fee154e487947177e5bd15363aa71b92f78f5a2a7145fdb81c5`
- `libsm6` deb SHA256: `1d27ebc381b499075a28c504be4d7d424c63b6866354736ac7f8d25813860cdf`

## Smoke Results

- Blender headless launch: pass
- `bpy` import: pass
- Blender version: `3.6.23`
- Blender embedded Python: `3.10.13`
- External env bridge: pass
- TensorFlow import from env inside Blender Python: `2.15.1`
- TFDS import from env inside Blender Python: `4.2.0`, `ReadWritePath=True`
- Official Google Research Kubric source import: pass
- `AssetSource.from_manifest`: present
- `kubric.simulator.PyBullet`: pass
- `kubric.renderer.Blender`: pass

## Caveats

- The `/home/hj/tools/void_kubric_exp58b/blender` symlink is not used for execution because it can break Blender's `$ORIGIN/lib` library resolution. Use the real binary path.
- Exp58B still needs Milestone D to audit the official script invocation and local manifest route before any render.
- Previous PAI TensorFlow GCS `gs://` access hung on metadata requests. If confirmed in Milestone D, use locally downloaded official manifest JSON files instead of `gs://` manifest paths.

## Safety

- Training run: no
- VOID inference run: no
- Preference forward / zero-gap / one-step: no
- 10-step: no
- Base/system env modified: no
- VOID official repo source modified: no
- `inference/metrics.py` modified: no
- Shared trainer modified: no
