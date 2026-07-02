# Exp58B Readback: Kubric Environment Blocker

Date: 2026-07-02

Status: `EXP58B_READBACK_DONE`

Storage status: `EXP58B_STORAGE_READY`

## Git

- Branch: `research/exp58-void-native-data-diagnostic-20260702`
- Starting HEAD: `80500de98864e5bee204f364865d53f379c0093b`
- Prior Exp58 status: `VOID_NATIVE_KUBRIC_BLOCKED`

## Answers

1. Failed imports in Exp58:
   - `kubric`: missing TensorFlow
   - `kubric.simulator.PyBullet`: missing TensorFlow
   - `kubric.renderer.Blender`: missing TensorFlow
   - `bpy`: missing module
   - `blender`: no executable found on PAI

2. `data_generation/kubric_variable_objects.py` is a normal Python script (`#!/usr/bin/env python3`) but imports `bpy` directly at top level. It also imports `kubric`, `PyBullet`, and `kubric.renderer.Blender` in the same Python process.

3. `bpy` is expected as Blender's embedded Python module or an equivalent Python package matching the active interpreter. PyPI did not provide a Python 3.10 `bpy` wheel in the previous probe, so Blender-embedded Python or a Blender-compatible package bridge is required.

4. A Blender executable is required by the official rendering path because the script uses `bpy` and `kubric.renderer.Blender`.

5. Python 3.10 remains the safest target because the existing PAI isolated env is Python 3.10.19 and TensorFlow CPU wheels are available for Python 3.10. A fresh env is required because the previous minimal env installed NumPy 2.2.6, which is not compatible with common TensorFlow 2.15-era constraints.

6. Candidate TensorFlow package: `tensorflow-cpu` in a fresh Python 3.10 environment. The install must pin compatible NumPy/Protobuf through the resolver; do not reuse the NumPy 2.2.6 minimal env.

7. PAI direct install may be possible for small packages, but previous PAI pip stalled on the `pybullet` wheel. For large TensorFlow/Blender assets, HAL/H20 wheelhouse or binary relay should be used if PAI download stalls.

8. If PAI cannot download TensorFlow or Blender reliably, relay wheels/binaries from HAL/H20 to PAI and verify SHA256. Do not leave large H20 staging behind.

9. No VOID inference, preference forward, zero-gap, one-step, or 10-step is allowed in Exp58B. This round only recovers the Kubric renderer environment and, if the env passes, renders Gate1/Gate8 native data.

## Storage Readback

Writable on PAI:

- `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/void/kubric_exp58b`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp58_void_native_data_diagnostic/exp58b_kubric_env`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp58_void_native_data_diagnostic/exp58b_kubric_env`

Not writable by `hj`:

- `/mnt/nas/hj/H20_Video_inpainting_DPO/tools/void_kubric_exp58b`
- `/mnt/nas/hj/conda_envs/void_kubric_exp58b`

Fallback roots if continuing:

- Env: `/home/hj/conda_envs/void_kubric_exp58b`
- Tools: `/home/hj/tools/void_kubric_exp58b`

Generated data should still be written under the writable NAS data/log/runtime roots.
