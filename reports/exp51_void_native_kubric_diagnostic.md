# Exp51 VOID-Native Kubric Diagnostic

Status: `VOID_NATIVE_KUBRIC_BLOCKED`

## Result

No VOID-native Kubric gate data was generated. This is a true environment/runtime blocker, not a data-quality result.

## Exact Blockers

- `KUBRIC_ENV_BLOCKED`: `kubric` and `pybullet` are not installed in the available Exp50 VOID environment.
- `KUBRIC_RUNTIME_BLOCKED`: `blender` is not available on H20 PATH, while official Kubric and HUMOTO renderers require Blender.
- `KUBRIC_ASSET_BLOCKED`: HUMOTO and Blender texture assets are absent. HUMOTO is license-gated/manual according to the official README.

GCS public Kubric asset manifests were reachable, but that is insufficient without Kubric/PyBullet/Blender. No fake VOID-native data was created.

## Probe

```text
HOST instance-afs92r3e
PYTHON /usr/bin/python3
IMPORT_FAIL kubric ModuleNotFoundError("No module named 'kubric'")
IMPORT_FAIL pybullet ModuleNotFoundError("No module named 'pybullet'")
IMPORT_OK imageio 2.37.0
IMPORT_OK imageio_ffmpeg 0.6.0
BLENDER_PATH None
BLENDER_FAIL FileNotFoundError(2, 'No such file or directory')
HUMOTO_DIR exists False
BLENDER_TEXTURE_DIR exists False
HTTP_OK KuBasic 200
HTTP_OK GSO 200
```

## Safety

No training, inference, GPU use, VOR-Eval, hard comp, official source modification, or synthetic fake replacement was performed.
