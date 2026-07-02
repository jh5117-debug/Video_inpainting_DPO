# Exp58 VOID Native Data Diagnostic Status

Current status: `VOID_NATIVE_KUBRIC_BLOCKED`

Storage status: `EXP58_STORAGE_PAI_NAS_PREFERRED`

Milestone A completed readback of Exp50-Exp57 and official VOID data-generation code. The requested PAI NAS experiment output root under `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo` is not writable by `hj`, but PAI logs/runtime and local `/home` are writable. H20 `/home/nvme01` has sufficient space for tiny smoke only.

Milestone B attempted the isolated Kubric environment smoke on PAI. Direct PAI pip stalled on `pybullet`, but a HAL wheelhouse relay succeeded and offline install completed. The environment remains blocked because `import kubric` requires TensorFlow and official VOID Kubric generation also requires Blender/`bpy`, neither of which is available in the isolated environment or PAI system path.

Final gate: blocked until a controlled TensorFlow-compatible Kubric environment and Blender/`bpy` renderer are available. No Kubric data, official Kubric inference, Kubric one-step, or 10-step was run.

## Exp58B

Current Exp58B status: `EXP58B_READBACK_DONE`.

Exp58B readback confirmed the exact renderer blocker and storage layout. PAI data/log/runtime roots are writable, while the preferred NAS tools/env roots are not writable by `hj`; isolated fallback roots under `/home/hj` are required for env/tools recovery. No training, inference, one-step, or 10-step has run.

Milestone B status: `EXP58B_KUBRIC_PYTHON_ENV_READY`.

The Python env `/home/hj/conda_envs/void_kubric_exp58b` now imports TensorFlow, TFDS 4.2.0, official Google Research Kubric source, `AssetSource.from_manifest`, PyBullet, image/video libraries, and OpenEXR/Imath. Blender/`bpy` remains pending for Milestone C. No render, inference, training, one-step, or 10-step has run.

Milestone C status: `EXP58B_BLENDER_BPY_READY`.

Official Blender 3.6.23 now runs headless from `/home/hj/tools/void_kubric_exp58b` with user-extracted `libSM/libICE` libraries. Blender Python imports `bpy` and can bridge the Exp58B env to import TensorFlow 2.15.1, TFDS 4.2.0, official source Kubric, `PyBullet`, and `kubric.renderer.Blender`. No render, VOID inference, training, one-step, or 10-step has run.

Milestone D status: `EXP58B_KUBRIC_INVOCATION_READY`.

An Exp58-only launcher now runs the unmodified official Kubric script under Blender Python with corrected `sys.argv`. Local official manifest JSONs load from NAS runtime storage, and a `--num_pairs 0` dry-run completed without rendering. Gate1 render smoke is next. No VOID inference, training, one-step, or 10-step has run.

Milestone E status: `VOID_NATIVE_KUBRIC_GATE1_READY`.

Gate1 generated one official Kubric paired sample with 24 frames at 128x128. The final env uses `OpenEXR==3.2.10`, an Exp58B launcher-only EXR channel shim, and static `imageio-ffmpeg`. Aggregate mask values include 0/63/127/255, and all Gate1 evidence sheets were opened. No VOID inference, training, one-step, or 10-step has run.

Final Exp58B status: `VOID_DATA_MISMATCH_TEST_READY`.

Gate8 generated 8/8 official Kubric native paired samples. All decode, all aggregate masks contain 0/63/127/255, and all review pages were opened. The manifest is `manifests/exp58b_void_native_kubric_gate8.jsonl`. No VOID inference, training, one-step, or 10-step has run.
