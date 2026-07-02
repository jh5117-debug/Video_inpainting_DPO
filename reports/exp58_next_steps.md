# Exp58 Next Steps

Minimal next action:

1. Prepare a controlled Kubric renderer environment on PAI/NAS or H20 with:
   - TensorFlow-compatible `kubric` import.
   - Blender executable or Python `bpy` available to the official script.
   - Existing minimal wheels retained for `pybullet`, `imageio`, `opencv`, and GCS.
2. Rerun only Exp58 Milestone B smoke.
3. If and only if B reaches `VOID_KUBRIC_ENV_READY`, generate tiny Kubric Gate8.
4. Visually review Gate8 before any inference or one-step.
5. Keep 10-step locked until a later one-step PASS exists.

Do not run another loss grid until the native-data diagnostic is available or explicitly abandoned.
