# Exp58B Kubric Python Environment

Date: 2026-07-02

Status: `EXP58B_KUBRIC_PYTHON_ENV_READY`

## Environment

- PAI host: `dsw-753014-85f54df947-bkp7h`
- Env path: `/home/hj/conda_envs/void_kubric_exp58b`
- Tools fallback: `/home/hj/tools/void_kubric_exp58b`
- Wheelhouse: `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp58_void_native_data_diagnostic/exp58b_kubric_env/tf_wheelhouse`
- Wheelhouse files: 94
- Wheelhouse SHA256: match
- Env size: about 2.1 GiB
- Kubric source checkout: `/home/hj/tools/void_kubric_exp58b/kubric_src`
- Kubric source commit: `61f2422c84bab75006df33c6989e0b483db3ccfe`

## Recovery Notes

The previous PyPI `kubric==0.1.1` wheel was not sufficient because it lacks `AssetSource.from_manifest`, which the official VOID `kubric_variable_objects.py` script calls. Exp58B installed the official Google Research Kubric source checkout as an editable package in the isolated env.

TensorFlow/TFDS required careful version control:

- `tensorflow-cpu==2.15.1`
- `tensorflow-datasets==4.2.0`
- `tensorflow-metadata==1.14.0`
- `protobuf==3.20.3`
- `numpy==1.26.4`

TFDS 4.9.4 failed because it removed the `ReadWritePath` API expected by Kubric. TFDS 4.2.0 restores `ReadWritePath`.

## Passing Imports

- `tensorflow 2.15.1`
- `tensorflow_datasets 4.2.0`
- `kubric HEAD` from source checkout
- `AssetSource.from_manifest`
- `kubric.simulator.PyBullet`
- `pybullet`
- `imageio 2.37.3`
- `cv2 4.11.0`
- `numpy 1.26.4`
- `OpenEXR`
- `Imath`

## Caveats

- `kubric.renderer.Blender` remains blocked by missing `bpy`. That is Milestone C.
- The official source `AssetSource.from_manifest` supports the required API, but HTTPS manifest paths raise `KeyError('https://')`. The default `gs://` path triggered TensorFlow GCS metadata hangs on PAI. Milestone D should use local downloaded manifest JSONs if Blender/`bpy` is fixed.
- `pip check` reports expected metadata caveats because Kubric's package metadata asks for `tensorflow`, `apache-beam[gcp]`, `cloudml-hypertune`, and `dataclasses`; this env intentionally uses `tensorflow-cpu` and does not install training/dataset-publishing extras.

## Decision

The TensorFlow/Kubric/PyBullet Python environment is recovered enough to proceed to Blender/`bpy` validation.
