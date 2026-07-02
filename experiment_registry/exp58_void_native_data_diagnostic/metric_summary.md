# Exp58 Metric Summary

No Exp58 Kubric generation, inference, or one-step metrics have been produced yet.

Kubric environment smoke:

- HAL wheelhouse: 44 files, 276 MiB local, 275 MiB on PAI.
- PAI wheelhouse SHA256: match.
- PAI isolated env size after install: about 937 MiB.
- Import OK: `numpy`, `imageio`, `cv2`, `pybullet`, `google.cloud.storage`.
- Import blocked: `kubric`, `kubric.simulator.PyBullet`, `kubric.renderer.Blender` due missing TensorFlow.
- Renderer blocked: `bpy` missing and no `blender` executable found.
- GCS default manifests: KuBasic, GSO, and HDRI Haven reachable.

Readback metrics:

- Exp57 H20 best diagnostic `ATS_STRICT_Q2_T500_S0`: full -0.094041, object -0.473428, overlap -0.581142, affected -0.146322, boundary -0.081617, outside -0.005963.
- Exp57 PAI best diagnostic `ATS_SDPO_Q2_T500_S0`: full +0.039160, object -0.337918, overlap -0.255698, affected +0.108109, boundary -0.049336, outside +0.075966.
- No Exp57 one-step PASS; 10-step remains locked.
