# Exp58 VOID Native Data Diagnostic

Date: 2026-07-02

Branch: `research/exp58-void-native-data-diagnostic-20260702`

Base: `origin/research/exp57-void-adaptive-transition-core-20260701`

## Goal

Exp58 tests whether VOID's remaining adapter failure is primarily a data-distribution mismatch rather than another loss tweak problem. Exp50-Exp57 established that VOID official inference, preference forward, zero-gap, one-step plumbing, and same-model loser generation work, but every VOR-derived one-step rescue remains mixed or negative because overlap, affected, and boundary regions regress.

## Hypothesis

VOID's official training data is generated paired counterfactual data, not VOR-derived object-removal data. The official release includes HUMOTO + Kubric generation code and `datasets/void_train_data.json` metadata, but not the rendered training videos in this repo checkout. VOR-derived quadmasks may not match VOID's native interaction-region semantics.

## Storage Policy

Large data should prefer PAI/NAS over H20 local storage. H20 `/home/nvme01` is allowed only for git worktree, isolated env, small staging, and short-lived render scratch.

Current audit:

- PAI `/home`: about 5T free.
- PAI `/mnt/nas`: very large and mounted.
- PAI requested experiment output parent `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo` is root-owned and not writable by `hj` for new Exp58 output root creation.
- PAI logs/runtime roots are writable.
- H20 `/home/nvme01`: about 1.2T free, 66% used; safe for tiny Gate8 smoke, not large accumulation.

## Allowed Work

- Add isolated code under `exp58_void_native_data_diagnostic/`.
- Set up an isolated Kubric env.
- Generate tiny Kubric Gate8 only if official dependencies and assets are available.
- Run official VOID inference and one-step diagnostics only if generated data is technically valid.

## Forbidden Work

- No 10-step or long training.
- No VOR-Eval.
- No hard comp.
- No VOID official source edits.
- No shared trainer or `inference/metrics.py` edits.
- No universal adapter, final SOTA, or third-backbone claim.

## Milestones

| Milestone | Status | Notes |
| --- | --- | --- |
| A | `EXP58_READBACK_DONE` | Data mismatch and storage readback completed. |
| Storage | `EXP58_STORAGE_PAI_NAS_PREFERRED` | PAI/NAS preferred; requested PAI experiment output root is not writable by `hj`. |
| B | `VOID_KUBRIC_ENV_BLOCKED` | PAI direct pip install stalled on `pybullet`; HAL wheelhouse relay succeeded, but `import kubric` still requires TensorFlow and official VOID Kubric generation requires missing Blender/`bpy`. |
| C | `VOID_NATIVE_KUBRIC_GATE8_BLOCKED` | Cannot generate official Kubric Gate8 without a working Kubric + Blender/`bpy` environment. |
| D | `VOID_KUBRIC_INFERENCE_BLOCKED` | No Kubric Gate8 exists. |
| E | `VOID_KUBRIC_ONESTEP_BLOCKED` | No Kubric native data exists. |
| F | `VOID_NATIVE_DATA_BLOCKED` | VOR-vs-Kubric comparison cannot be computed until native data exists. |
| G | `VOID_NATIVE_KUBRIC_BLOCKED` | Final decision: data-mismatch hypothesis remains untested because official native data generation is environment-blocked. |

## Scientific Position

VOID remains a VOR-OR inference baseline, same-model loser-generator candidate, and adapter-engineering candidate. Exp58 may support `VOID_DATA_MISMATCH_SUSPECTED` or confirm a native-data path, but it cannot claim third-backbone evidence without a later one-step PASS and aggregator-approved 10-step positive result.

## Milestone B Result

Status: `VOID_KUBRIC_ENV_BLOCKED`.

The PAI isolated env is `/home/hj/conda_envs/void_kubric_exp58`. Direct PAI pip install stalled for more than 17 minutes on the `pybullet` wheel, so a HAL wheelhouse relay was used instead. The wheelhouse contains 44 files, is 275 MiB on PAI, and passed SHA256 verification.

Offline install into the isolated env succeeded for the minimal runtime packages (`pybullet`, `imageio`, `opencv-python-headless`, `numpy`, `google-cloud-storage`, and related wheels). GCS access to the official Kubric default manifests succeeded for KuBasic, GSO, and HDRI Haven.

The generator is still blocked:

- `import kubric` fails with `ModuleNotFoundError("No module named 'tensorflow'")`.
- `from kubric.simulator import PyBullet` and `from kubric.renderer import Blender` fail for the same missing TensorFlow dependency.
- `import bpy` fails with `ModuleNotFoundError("No module named 'bpy'")`.
- No `blender` executable was found on PAI.

Because the official VOID Kubric script imports `bpy` directly and uses `kubric.renderer.Blender`, Exp58 cannot generate valid VOID-native Kubric data in the current environment. No fake data was created.

## Final Decision

Status: `VOID_NATIVE_KUBRIC_BLOCKED`.

Exp58 confirmed that the official Kubric data path is the right native-data diagnostic target, but it did not produce native data. The blocker is exact and reproducible: the isolated PAI environment cannot import Kubric without TensorFlow, and the official renderer cannot run without Blender/`bpy`. The official public GCS assets are reachable, so this is not an asset-network blocker.

No Kubric Gate8, official Kubric inference, Kubric one-step, VOR-vs-Kubric quantitative comparison, training, or 10-step was run. VOID remains a VOR-OR inference baseline, same-model loser generator, and adapter-engineering candidate, not third-backbone evidence.

## Exp58B Environment Recovery

Exp58B continues on the same branch and targets only Kubric renderer recovery. It does not run VOID inference, preference forward, zero-gap, one-step, 10-step, or loss tuning.

Milestone A status: `EXP58B_READBACK_DONE`; storage status: `EXP58B_STORAGE_READY`.

Confirmed blocker:

- `kubric_variable_objects.py` is invoked as a normal Python script, but it imports `bpy` at top level.
- The same process must also import `kubric`, `PyBullet`, and `kubric.renderer.Blender`.
- Existing PAI minimal env has `pybullet`, `imageio`, `cv2`, `numpy`, and GCS imports, but `kubric` fails because TensorFlow is missing.
- PAI has no system `blender` executable and no `bpy` module.

Storage update:

- PAI data/log/runtime roots for Exp58B are writable.
- PAI NAS tools root `/mnt/nas/hj/H20_Video_inpainting_DPO/tools/void_kubric_exp58b` is not writable by `hj`.
- PAI NAS env root `/mnt/nas/hj/conda_envs/void_kubric_exp58b` is not writable by `hj`.
- Exp58B must use `/home/hj/conda_envs/void_kubric_exp58b` and `/home/hj/tools/void_kubric_exp58b` as isolated fallbacks if continuing.

Candidate Python route: Python 3.10 isolated env with `tensorflow-cpu` and Kubric 0.1.1, plus a controlled Blender/`bpy` bridge. Because TensorFlow wheels are sensitive to NumPy/Protobuf versions, Exp58B should use a fresh env rather than mutate the previous minimal env.

Milestone B status: `EXP58B_KUBRIC_PYTHON_ENV_READY`.

The Python side was recovered in a fresh env at `/home/hj/conda_envs/void_kubric_exp58b` using a HAL-built wheelhouse relayed to PAI runtime storage. PyPI `kubric==0.1.1` was insufficient because it lacks `AssetSource.from_manifest`; Exp58B therefore installed the official Google Research Kubric source checkout (`61f2422c84bab75006df33c6989e0b483db3ccfe`) into the isolated env without modifying VOID official source.

Working imports:

- `tensorflow==2.15.1`
- `tensorflow_datasets==4.2.0` with `ReadWritePath`
- official source Kubric with `AssetSource.from_manifest`
- `kubric.simulator.PyBullet`
- `pybullet`
- `imageio`
- `opencv-python-headless`
- `numpy==1.26.4`
- `OpenEXR` / `Imath`

Caveats:

- `kubric.renderer.Blender` remains blocked by missing `bpy`; this is Milestone C.
- `AssetSource.from_manifest` does not accept HTTPS manifest URLs, while `gs://` access via TensorFlow GCS metadata requests hangs on PAI. Milestone D should use local downloaded manifest JSON files if Blender/`bpy` is fixed.

Milestone C status: `EXP58B_BLENDER_BPY_READY`.

The renderer side was recovered with official Blender 3.6.23 under `/home/hj/tools/void_kubric_exp58b`. The preferred NAS tools root remains unwritable by `hj`, so this is an isolated `/home/hj` tools fallback rather than a system install. Blender required `libSM.so.6` and `libICE.so.6`; Exp58B downloaded the Ubuntu Noble `libsm6` and `libice6` deb packages and extracted them under the tools root, using `LD_LIBRARY_PATH` only for this Blender invocation.

Validated smoke:

- Blender headless launch: pass.
- `bpy` import: pass.
- Blender embedded Python: `3.10.13`.
- TensorFlow bridge from `/home/hj/conda_envs/void_kubric_exp58b`: `2.15.1`.
- TFDS bridge: `4.2.0` with `ReadWritePath`.
- Official source Kubric bridge: pass, including `AssetSource.from_manifest`.
- `kubric.simulator.PyBullet` and `kubric.renderer.Blender`: pass.

No Kubric render, VOID inference, preference forward, one-step, or 10-step was run. Milestone D must audit the exact official script invocation before Gate1.

Milestone D status: `EXP58B_KUBRIC_INVOCATION_READY`.

Directly running `blender -b --python kubric_variable_objects.py -- ...` causes the script parser to see Blender's own CLI arguments. Exp58B added an isolated launcher at `exp58_void_native_data_diagnostic/exp58b_kubric_env_recovery/blender_void_kubric_launcher.py` that leaves the official VOID script unchanged, injects the Kubric/TensorFlow env into Blender Python, replaces `sys.argv`, and executes the script with `runpy`.

The three public official manifest JSON files were cached in NAS runtime storage and load locally:

- KuBasic: 15 assets.
- GSO: 1033 assets.
- HDRI Haven: 509 assets.

A no-render `--num_pairs 0` dry-run through Blender Python completed successfully, creating only the dry-run output directory. Gate1 should use the same launcher, local manifests, `--num_pairs 1`, and a low-resolution `--fast` smoke. If per-asset GCS tarball fetching fails during Gate1, mark that exact blocker; do not fake data.

Milestone E status: `VOID_NATIVE_KUBRIC_GATE1_READY`.

Gate1 generated exactly one official Kubric pair at `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/void/kubric_exp58b/gate1/gate1/00000`. The first render attempt exposed an `OpenEXR==3.4.13` segfault during Kubric EXR postprocessing; Exp58B fixed this by using `OpenEXR==3.2.10` in the isolated env plus a launcher-only channel-name shim for Blender's uppercase EXR channels. A second attempt exposed broken PAI system ffmpeg (`libblas.so.3` missing); Exp58B fixed this by placing the static `imageio-ffmpeg` binary first on `PATH`.

The final Gate1 sample decodes as 24 frames at 128x128/8fps. Aggregate mask values include 0/63/127/255. All visual evidence sheets were opened. This is a renderer smoke pass and not a data-mismatch or adapter-quality conclusion. Gate8 generation is now allowed, but only N=8 and still no VOID inference/training in Exp58B.

Milestone F/G status: `VOID_DATA_MISMATCH_TEST_READY`.

Gate8 generated exactly 8 native Kubric pairs under `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/void/kubric_exp58b/gate8/gate8`. All samples decode as 24 frames at 128x128/8fps, and every aggregate mask contains values 0/63/127/255. `manifests/exp58b_void_native_kubric_gate8.jsonl` records `rgb_full`, `rgb_altered_physics` as the removed/altered winner path, `mask.mp4`, metadata, areas, generation seeds, and review pages.

All eight review pages were opened. Caveat: every metadata file reports `target_hit=false`; the samples remain useful for native-data environment and initial VOR-vs-Kubric distribution diagnostics, but they do not establish adapter evidence. Exp58B stops before VOID inference, preference forward, one-step, or 10-step.
