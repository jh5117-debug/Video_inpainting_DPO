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

Exp58B renderer smoke:

- Blender version: 3.6.23.
- Blender embedded Python: 3.10.13.
- User-shared libraries: `libSM.so.6` and `libICE.so.6` extracted under `/home/hj/tools/void_kubric_exp58b/user_libs`.
- Blender Python import OK: `bpy`, TensorFlow 2.15.1, TFDS 4.2.0, official source Kubric, `PyBullet`, and `kubric.renderer.Blender`.
- No render metrics yet; Gate1 has not run.

Exp58B invocation audit:

- Local official manifests: KuBasic 15 assets, GSO 1033 assets, HDRI Haven 509 assets.
- Blender Python dry-run: `--num_pairs 0`, pass.
- Output path convention: `job_dir/out_prefix/00000` for real samples.
- No native-data video metrics yet.

Exp58B Gate1 render:

- Samples: 1.
- Frames/resolution/fps: 24 frames, 128x128, 8fps.
- Aggregate mask counts: 0=1312, 63=420, 127=11797, 255=379687.
- Metadata: 3 objects, 1 removed, `target_hit=false`, camera motion `zoom_in`.
- Compatibility fixes: `OpenEXR==3.2.10`, launcher-only EXR channel shim, static `imageio-ffmpeg`.

Readback metrics:

- Exp57 H20 best diagnostic `ATS_STRICT_Q2_T500_S0`: full -0.094041, object -0.473428, overlap -0.581142, affected -0.146322, boundary -0.081617, outside -0.005963.
- Exp57 PAI best diagnostic `ATS_SDPO_Q2_T500_S0`: full +0.039160, object -0.337918, overlap -0.255698, affected +0.108109, boundary -0.049336, outside +0.075966.
- No Exp57 one-step PASS; 10-step remains locked.
- Exp58 produced no native-data metrics because Kubric generation was environment-blocked.
