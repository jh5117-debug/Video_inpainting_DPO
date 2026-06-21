# Exp23: Two-Stage Pool Morphology Sweep

Status: `TRAINER_WIRED_PHY_LAUNCH_PENDING`

Branch: `research/exp23-two-stage-pool-morphology-sweep`

HAL worktree: `/home/hj/H20_Video_inpainting_DPO_exp23_pool_sweep`

PAI worktree target: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp23_pool_sweep`

## Motivation

Exp20 found that broad image-space boundary tuning produced small search-dev signals but did not survive multi-seed shadow-dev confirmation. Exp23 restarts this line with a more faithful DiffuEraser setting:

- full Stage1 2000 + Stage2 2000 training;
- fresh Exp11 twin for every candidate/seed;
- DAVIS50 explicitly used as `DAVIS50_MODEL_SELECTION_BENCHMARK`;
- independent Stage1/Stage2 inner and outer max-pool morphology.

## Morphology Definition

Mask convention: `M=1` means hole/unknown.

The public geometry parameters are:

- `pool_grid_scale`
- `inner_pool_steps`
- `outer_pool_steps`
- `inner_weight`
- `outer_weight`

No candidate is named by image-space pixel radius.

One pool step is:

```python
F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
```

Regions:

- `mask_core = erode_steps(M, inner_pool_steps)`
- `inner_ring = M - mask_core`
- `outer_ring = dilate_steps(M, outer_pool_steps) - M`
- `far_outside = 1 - mask_core - inner_ring - outer_ring`

`pool_grid_scale=1` is legacy latent/loss-grid pooling. `pool_grid_scale=2/4` upsamples the binary latent mask using nearest, builds the partition on the finer pool grid, then area-pools back to loss grid.

Exp11 exact setting:

- `pool_grid_scale=1`
- Stage1: `inner=0`, `outer=1`, `inner_weight=0`, `outer_weight=0.75`
- Stage2: `inner=0`, `outer=1`, `inner_weight=0`, `outer_weight=0.75`
- `mask_core_weight=1.0`
- `outside_weight=0.05`

## Aggregation

Two aggregation modes are implemented:

- `legacy_weighted`
- `region_balanced`

The first parity target is `legacy_weighted` with Exp11 exact morphology.

## Data

Training manifest target:

`/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/exp09_10_11_youtubevos_gtwin_d3comp_pai/manifests/selected_primary_comp.gtwin.pai_paths.jsonl`

This path must be revalidated on PAI before training.

DAVIS50 target:

`/mnt/workspace/hj/nas_hj/data/external/davis_432_240`

## Current Implementation

Implemented:

- `pool_morphology.py`
- `region_aggregation.py`
- `process_title.py`
- initial morphology and process-title tests
- GPU4-7 release audit

Not yet launched:

- Stage1/Stage2 trainer copy;
- Exp11 full loss parity;
- PAI DDP training;
- DAVIS50 paired evaluation.

## GPU Status

GPU4-7 initial audit result: `BLOCKED_GPU4_7_NOT_AVAILABLE`.

GPUs 4-6 are occupied by active `/mnt/workspace/xiaoqi/multigen/...` training processes. GPU7 has 58GB NVML memory with no live `/proc` PID. No process was killed.

See:

- `reports/exp23_gpu4_7_release_audit.md`
- `reports/exp23_gpu4_7_release_audit.csv`

## Next Gate

Do not start Exp23 training until GPU4-7 are safely available or the user/admin confirms ownership and termination permission for the listed processes.

## 2026-06-21 Force-Release Attempt

The user explicitly authorized terminating all user/collaborator jobs on GPU4-7.

Actions completed on PAI:

- saved pre-release state;
- identified GPU4-6 high-expert multigen launcher/worker set;
- sent `TERM` to high launcher `1246572` and worker process groups `-1246702`, `-1246703`, `-1246704`;
- confirmed GPU4-6 memory dropped to idle-level usage;
- probed GPU7's persistent `[Not Found]` NVML context.

Result:

| GPU | before used MiB | after used MiB | status |
|---:|---:|---:|---|
| 4 | 141333 | 244 | released |
| 5 | 141045 | 4 | released |
| 6 | 142321 | 292 | released |
| 7 | 58071 | 58071 | not released: PID `1758887` has no `/proc` entry |

GPU7 remains a ghost allocation. Per the user's safety constraints, no GPU reset or server restart was used.

Training was not launched because the current Exp23 branch still lacks the real isolated Stage1 trainer, Stage2 trainer, paired queue/controller, and DAVIS50 evaluator. The current PAI launch script is a blocked placeholder, so launching it would not run the requested two-stage sweep.

Reports:

- `reports/exp23_gpu4_7_force_release_audit.md`
- `reports/exp23_gpu4_7_force_release_audit.csv`

## 2026-06-21 Trainer Wiring / Phy Relaunch Update

Added isolated Exp23 training plumbing:

- `exp23_two_stage_pool_morphology_sweep/code/train_stage1.py`
- `exp23_two_stage_pool_morphology_sweep/code/train_exp23_stage1.py`
- `exp23_two_stage_pool_morphology_sweep/code/train_exp23_stage2.py`
- `exp23_two_stage_pool_morphology_sweep/code/exp23_trial_runner.py`
- `exp23_two_stage_pool_morphology_sweep/scripts/launch_exp23_phy_sweep_pai.sh`

Local validation passed:

- `python -m py_compile exp23_two_stage_pool_morphology_sweep/code/*.py`
- `python -m unittest discover -s exp23_two_stage_pool_morphology_sweep/tests -p 'test_*.py'`
- `bash -n exp23_two_stage_pool_morphology_sweep/scripts/*.sh`
- `git diff --check`

First PAI launch target:

```text
pair_id = phaseA_scale1_pair001_outer2
fresh Exp11 twin = legacy exact, pool_grid_scale=1, inner=0, outer=1, inner_weight=0, outer_weight=0.75
candidate = pool_grid_scale=1, inner=0, outer=2, inner_weight=0, outer_weight=0.75
CUDA_VISIBLE_DEVICES=4,5,6,7
nproc_per_node=4
```

GPU7 still has a persistent `[Not Found]` NVML allocation, so the Phy launch is expected to be a real four-GPU test rather than a guaranteed successful run. No three-GPU fallback and no GPU reset will be used.

## 2026-06-21 PAI Phy Launch Result

PAI was fast-forwarded to:

```text
d9d7077c281af33e7186f890d5175e4d470c1d8b
```

Launch command:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash exp23_two_stage_pool_morphology_sweep/scripts/launch_exp23_phy_sweep_pai.sh
```

Runtime process identity was valid:

```text
controller PID = 1285825, comm=Phy, exe=/mnt/nas/hj/conda_envs/diffueraser/bin/Phy
torchrun PID   = 1285828, exe=/mnt/nas/hj/conda_envs/diffueraser/bin/Phy
rank PIDs      = 1285905, 1285906, 1285907, 1285908, exe=/mnt/nas/hj/conda_envs/diffueraser/bin/Phy
```

The training was real enough to complete one optimizer step and write `dpo_diagnostics.csv`:

```text
global_step=1
total_loss=0.698884
dpo_loss=0.693339
grad_norm=2.886552
implicit_acc=0.25
loser_dominant_ratio=0.0
```

Failure:

```text
rank3 / local_rank3 OOM on GPU7
Process 1758887 has 56.70 GiB memory in use
```

GPU7's stale no-proc allocation is therefore the active blocker. The Phy workers exited cleanly after torch distributed propagated the rank3 failure. No Exp23 worker remains running.

## 2026-06-21 GPU2/4/5/6 Retry

Per user instruction, the sweep was restarted without GPU7:

```bash
CUDA_VISIBLE_DEVICES=2,4,5,6 \
PAIR_ID=phaseA_scale1_pair001_outer2_gpus2456 \
LOG_PATH=logs/pipelines/exp23_phy_sweep_controller_gpus2456.log \
PID_PATH=exp23_two_stage_pool_morphology_sweep/runtime/exp23_phy_sweep_controller_gpus2456.pid \
bash exp23_two_stage_pool_morphology_sweep/scripts/launch_exp23_phy_sweep_pai.sh
```

Current state:

```text
status = RUNNING
controller PID = 1289732
torchrun PID = 1289735
rank PIDs = 1289812, 1289813, 1289814, 1289815
GPU mapping = 2,4,5,6
pair_id = phaseA_scale1_pair001_outer2_gpus2456
current model = fresh_exp11_outer_b075
```

15-minute monitor result:

```text
Stage1 step >= 170
latest logged total_loss ~= 0.628861
latest logged dpo_loss ~= 0.625165
latest logged grad_norm ~= 2.845862
no OOM / no Traceback
all four active CUDA processes show process name Phy
```

Runtime monitor:

```text
monitor PID = 1291494
monitor log = exp23_two_stage_pool_morphology_sweep/runtime/monitor_gpus2456.log
```

Risk note: loser-dominant diagnostics are high during early fresh Exp11 Stage1, e.g. `loser_dominant_ratio=1.0` at steps 40-170. This is a training diagnostic to watch, not a launch blocker.
