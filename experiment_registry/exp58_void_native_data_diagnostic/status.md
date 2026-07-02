# Exp58 VOID Native Data Diagnostic Status

Current status: `VOID_KUBRIC_ENV_BLOCKED`

Storage status: `EXP58_STORAGE_PAI_NAS_PREFERRED`

Milestone A completed readback of Exp50-Exp57 and official VOID data-generation code. The requested PAI NAS experiment output root under `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo` is not writable by `hj`, but PAI logs/runtime and local `/home` are writable. H20 `/home/nvme01` has sufficient space for tiny smoke only.

Milestone B attempted the isolated Kubric environment smoke on PAI. Direct PAI pip stalled on `pybullet`, but a HAL wheelhouse relay succeeded and offline install completed. The environment remains blocked because `import kubric` requires TensorFlow and official VOID Kubric generation also requires Blender/`bpy`, neither of which is available in the isolated environment or PAI system path.

Next gate: blocked until a controlled Blender/`bpy` + TensorFlow-compatible Kubric environment is provided.
