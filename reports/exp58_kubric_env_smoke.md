# Exp58 Kubric Environment Smoke

Date: 2026-07-02

Status: `VOID_KUBRIC_ENV_BLOCKED`

## Environment

- Machine used for PAI smoke: `dsw-753014-85f54df947-bkp7h`
- PAI isolated env: `/home/hj/conda_envs/void_kubric_exp58`
- Python: 3.10.19
- VOID repo inspected: `/mnt/nas/hj/H20_Video_inpainting_DPO/third_party/VOID/Netflix_void-model`
- Logs: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp58_void_native_data_diagnostic/`
- Runtime wheelhouse: `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp58_void_native_data_diagnostic/kubric_wheelhouse`

## Install Attempts

Direct PAI pip install of full `kubric` dependencies was not usable: it backtracked through old dependencies and failed while building an old `absl-py` path. A second minimal PAI install attempt stalled for more than 17 minutes on `pybullet-3.2.7` download and was stopped by terminating only the Exp58 Kubric install process group.

HAL then built a Python 3.10 wheelhouse and relayed it to PAI:

- Wheelhouse files: 44
- HAL wheelhouse size: 276 MiB
- PAI wheelhouse size: 275 MiB
- SHA256 verification: match

Offline install from the wheelhouse succeeded for the minimal package set:

- `kubric==0.1.1` installed with `--no-deps`
- `pybullet`
- `imageio`
- `imageio-ffmpeg`
- `opencv-python-headless`
- `numpy`
- `tqdm`
- `pillow`
- `google-cloud-storage`
- `pandas`
- `pyquaternion`
- `trimesh`
- `munch`
- `pypng`
- `traitlets`
- `bidict`
- `singledispatchmethod`
- `scikit-learn`

Pip correctly reported that the full `kubric` metadata dependencies were intentionally not installed: `apache-beam[gcp]`, `cloudml-hypertune`, `tensorflow`, and `tensorflow-datasets`.

## Import Smoke

Passing imports:

- `numpy 2.2.6`
- `imageio 2.37.3`
- `cv2 4.13.0`
- `pybullet`
- `google.cloud.storage`

Blocking imports:

- `kubric`: `ModuleNotFoundError("No module named 'tensorflow'")`
- `kubric.simulator.PyBullet`: `ModuleNotFoundError("No module named 'tensorflow'")`
- `kubric.renderer.Blender`: `ModuleNotFoundError("No module named 'tensorflow'")`
- `bpy`: `ModuleNotFoundError("No module named 'bpy'")`

No `blender` executable was found on PAI.

## GCS Asset Probe

The official default Kubric manifests are reachable:

- `https://storage.googleapis.com/kubric-public/assets/KuBasic/KuBasic.json`: HTTP 200
- `https://storage.googleapis.com/kubric-public/assets/GSO/GSO.json`: HTTP 200
- `https://storage.googleapis.com/kubric-public/assets/HDRI_haven/HDRI_haven.json`: HTTP 200

## Decision

The blocker is environment-side, not data-fabrication or model-side:

`VOID_KUBRIC_ENV_BLOCKED_TENSORFLOW_AND_BLENDER_BPY_MISSING`

Official VOID Kubric generation cannot run because the script imports `bpy` directly and uses `kubric.renderer.Blender`. Installing minimal Python wheels is insufficient without a controlled TensorFlow-compatible Kubric install plus Blender/`bpy` availability. No Kubric videos were generated and no fake native data was created.
