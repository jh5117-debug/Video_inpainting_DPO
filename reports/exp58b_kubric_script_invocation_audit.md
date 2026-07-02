# Exp58B Kubric Script Invocation Audit

Status: `EXP58B_KUBRIC_INVOCATION_READY`

## Scope

This milestone audited the official VOID Kubric generation invocation and ran only a no-render `--num_pairs 0` dry-run. It did not render Kubric Gate1, run VOID inference, train, run preference forward, run one-step, or run 10-step.

## Official Script

- Path: `/mnt/nas/hj/H20_Video_inpainting_DPO/third_party/VOID/Netflix_void-model/data_generation/kubric_variable_objects.py`
- Entrypoint: normal Python script with `main()`
- Top-level imports: `bpy`, `kubric`, `PyBullet`, `kubric.renderer.Blender`, `pybullet`
- Required runtime: Blender Python or equivalent working `bpy` runtime

## Invocation Detail

Direct invocation as:

```bash
blender -b --python kubric_variable_objects.py -- ...
```

is not safe because the script calls `parser.parse_args()` with no explicit args, so it can see Blender's own `-b --python ...` arguments. Exp58B therefore uses a thin launcher:

`exp58_void_native_data_diagnostic/exp58b_kubric_env_recovery/blender_void_kubric_launcher.py`

The launcher:

- adds `/home/hj/tools/void_kubric_exp58b/kubric_src` to Blender Python `sys.path`;
- adds `/home/hj/conda_envs/void_kubric_exp58b/lib/python3.10/site-packages` to `sys.path`;
- replaces `sys.argv` with `[official_script] + official_args`;
- executes the official script with `runpy.run_path(..., run_name="__main__")`.

## Arguments

Script arguments confirmed by `--help`:

- `--job-dir`
- `--scratch_dir`
- `--objects_split {train,test}`
- `--backgrounds_split {train,test}`
- `--kubasic_assets`
- `--gso_assets`
- `--hdri_assets`
- `--out_prefix`
- `--num_pairs`
- `--start_index`
- `--fast`
- inherited Kubric args including `--frame_rate`, `--step_rate`, `--frame_start`, `--frame_end`, `--seed`, and `--resolution`

Output structure:

`job_dir/out_prefix/00000/`

for rendered samples. With `--num_pairs 0`, only `job_dir/out_prefix` is created.

## Local Manifest Route

`AssetSource.from_manifest()` can read local official manifest JSON files. Exp58B cached the public official manifests in NAS runtime storage:

- `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp58_void_native_data_diagnostic/exp58b_kubric_env/manifests/KuBasic.json`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp58_void_native_data_diagnostic/exp58b_kubric_env/manifests/GSO.json`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp58_void_native_data_diagnostic/exp58b_kubric_env/manifests/HDRI_haven.json`

Manifest SHA256:

- `KuBasic.json`: `00e966d88a0058b57bf5854ebe84b2806678656f6e96f892777d8d49d3b5d2f3`
- `GSO.json`: `6b3a738ca198a78e146b751cae21e45cc12c69f16561dedf068b453c1be15cb0`
- `HDRI_haven.json`: `6407c2655633023576352c36d309d4fb1cca241ab01a776463740e0e7d5879c1`

Manifest inventory:

- KuBasic: 15 assets
- GSO: 1033 assets
- HDRI Haven: 509 assets

The manifest `data_dir` values remain `gs://kubric-public/assets/...`, so actual Gate1 render will still fetch per-asset tarballs from official GCS via TensorFlow file IO. The prior `gs://` metadata hang was for manifest loading; using local manifests avoids that specific failure.

## No-Render Dry-Run

Command class:

```bash
PYTHONNOUSERSITE=1 \
LD_LIBRARY_PATH=/home/hj/tools/void_kubric_exp58b/user_libs/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
/home/hj/tools/void_kubric_exp58b/blender-3.6.23-linux-x64/blender \
  -b --python exp58_void_native_data_diagnostic/exp58b_kubric_env_recovery/blender_void_kubric_launcher.py -- \
  --job-dir /mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp58_void_native_data_diagnostic/exp58b_kubric_env/dryrun \
  --scratch_dir /mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp58_void_native_data_diagnostic/exp58b_kubric_env/scratch_dryrun \
  --kubasic_assets /mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp58_void_native_data_diagnostic/exp58b_kubric_env/manifests/KuBasic.json \
  --gso_assets /mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp58_void_native_data_diagnostic/exp58b_kubric_env/manifests/GSO.json \
  --hdri_assets /mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp58_void_native_data_diagnostic/exp58b_kubric_env/manifests/HDRI_haven.json \
  --out_prefix exp58b_dryrun \
  --num_pairs 0 \
  --resolution 128 \
  --fast \
  --seed 580
```

Result:

- Official script loaded all three manifests.
- Dry-run completed with `Successful: 0/0`, `Failed: 0/0`.
- Output directory created: `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp58_void_native_data_diagnostic/exp58b_kubric_env/dryrun/exp58b_dryrun`
- No rendered videos or fake native data were created.

## Gate1 Plan

Gate1 should use the same launcher and local manifests, with:

- `--num_pairs 1`
- `--start_index 0`
- `--out_prefix gate1`
- `--job-dir /mnt/nas/hj/H20_Video_inpainting_DPO/data/external/void/kubric_exp58b/gate1`
- `--scratch_dir /mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp58_void_native_data_diagnostic/exp58b_kubric_env/scratch_gate1`
- `--fast`
- low resolution first, preferably `128` or `256`

If asset tarball fetching from `gs://kubric-public/assets/...` fails or hangs during Gate1, mark Gate1 blocked with the exact GCS asset fetch blocker rather than faking data.

## Safety

- Render run: no
- VOID inference run: no
- Training run: no
- Preference forward / zero-gap / one-step: no
- 10-step: no
- VOID official repo source modified: no
- Base/system env modified: no
