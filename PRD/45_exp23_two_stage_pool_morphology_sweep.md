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

500-step monitor result:

```text
timestamp = 2026-06-21 08:52 CST
Stage1 step >= 510
checkpoint-500 = written
latest logged total_loss ~= 0.573817
latest logged dpo_loss ~= 0.570924
latest logged grad_norm ~= 10.205749
no OOM / no Traceback / no NaN grep hits
all four active CUDA processes still show process name Phy
```

## 2026-06-21 GPU2/4/5/6 Pair Completion

The first Phase A paired training run completed on PAI after switching from
GPU4/5/6/7 to GPU2/4/5/6.

```text
pair_id = phaseA_scale1_pair001_outer2_gpus2456
fresh Exp11 twin = fresh_exp11_outer_b075
candidate = candidate_scale1_outer2_b075
GPU mapping = 2,4,5,6
started_at = 2026-06-21 11:43:57 CST
finished_at = 2026-06-21 19:47:50 CST
queue_status = STAGE1_STAGE2_PAIR_COMPLETED
```

All four training stages completed 2000 optimizer steps and wrote checkpoints
plus `last_weights`:

| model | stage | last step | final total loss | final dpo loss | final grad norm | max grad norm | last loser-dominant |
|---|---:|---:|---:|---:|---:|---:|---:|
| fresh_exp11_outer_b075 | Stage1 | 2000 | 0.399996 | 0.392117 | 222.153356 | 728.881603 | 1.000000 |
| fresh_exp11_outer_b075 | Stage2 | 2000 | 0.404275 | 0.392957 | 7.555776 | 63.415727 | 1.000000 |
| candidate_scale1_outer2_b075 | Stage1 | 2000 | 0.432572 | 0.428459 | 74.767941 | 578.498537 | 1.000000 |
| candidate_scale1_outer2_b075 | Stage2 | 2000 | 0.498125 | 0.489977 | 7.075422 | 30.997587 | 1.000000 |

Output root:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp23_two_stage_pool_morphology_sweep/pairs/phaseA_scale1_pair001_outer2_gpus2456/
```

Candidate final weights:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp23_two_stage_pool_morphology_sweep/pairs/phaseA_scale1_pair001_outer2_gpus2456/candidate_scale1_outer2_b075/stage2/last_weights
```

Fresh Exp11 twin final weights:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp23_two_stage_pool_morphology_sweep/pairs/phaseA_scale1_pair001_outer2_gpus2456/fresh_exp11_outer_b075/stage2/last_weights
```

Final GPU2/4/5/6 state after training:

```text
GPU2 = 0 MiB, 0%
GPU4 = 244 MiB, 0%
GPU5 = 4 MiB, 0%
GPU6 = 292 MiB, 0%
```

No Exp23 `Phy` process remained after completion.

Risk notes:

- Both Stage1 runs had large finite gradient spikes; the largest observed
  values were `728.88` for the fresh Exp11 twin and `578.50` for the
  candidate. Stage2 max gradients were lower but still nontrivial.
- Final loser-dominant ratios remained `1.0` for all four stages.
- The current runner completed paired Stage1+Stage2 training only. DAVIS50
  evaluation was not launched by this runner and must remain the next gate
  before making any method-quality claim.

Report:

- `reports/exp23_gpu2456_pair_completion_report.md`

## 2026-06-21 Pair001 Boundary Audit

Status:

```text
PAIR001_CONTROL_INVALID_BOUNDARY_MODE
```

The completed `phaseA_scale1_pair001_outer2_gpus2456` pair cannot be used for
scientific outer1-vs-outer2 comparison. Runtime `dpo_diagnostics.csv` evidence
shows the fresh Exp11 control used:

```text
fresh Stage1 boundary_mode = both
fresh Stage2 boundary_mode = both
```

The root cause was the legacy exact path reading `BOUNDARY_MODE` with a silent
default of `both`, while the Exp23 runner did not explicitly pass or record
`boundary_mode=outer`.

Corrective code change:

- Stage1 and Stage2 now require explicit `--boundary_mode`.
- The runner passes `--boundary_mode outer` for both fresh Exp11 and the outer2
  candidate.
- Corrected runs write `resolved_region_config.json` and `region_diagnostics.csv`.

Next action:

```text
rerun pair_id = phaseA_scale1_pair001_outer2_corrected_outer_control_seed20260619_gpus2456
```

No DAVIS50 comparison should be made on the invalid-control pair.

## 2026-06-21 Corrected Pair Rerun Started

Status:

```text
PAIR001_CORRECTED_OUTER_CONTROL_RERUN_RUNNING
```

Corrected pair:

```text
pair_id = phaseA_scale1_pair001_outer2_corrected_outer_control_seed20260619_gpus2456
gpus = 2,4,5,6
controller_pid = 1428304
torchrun_pid = 1428307
rank_pids = 1428380,1428381,1428382,1428383
process_name = Phy
PAI_HEAD = 2e1988c77e43b10cadc7ed8c19b1eda53d8e8a55
```

GPU2/4/5/6 were released from a targeted
`/mnt/workspace/xiaoqi/multigen/.../qxq_sample_base_dense_v0.py` process group
before launch. GPU0/1/3 tasks were not touched, and GPU7 remains excluded due
to the persistent NVML ghost allocation.

Runtime evidence for the fresh Exp11 control is now explicit:

```json
{
  "legacy_exact": true,
  "boundary_mode": "outer",
  "pool_grid_scale": 1,
  "inner_pool_steps": 0,
  "outer_pool_steps": 1,
  "mask_region_weight": 1.0,
  "boundary_region_weight": 0.75,
  "outside_region_weight": 0.05,
  "aggregation": "legacy_global_weighted_mean"
}
```

The fresh Stage1 `region_diagnostics.csv` and `dpo_diagnostics.csv` both record
`boundary_mode=outer` through at least step 140, so the corrected rerun is
currently producing valid runtime evidence. No next morphology candidate should
be launched until this corrected pair finishes Stage1 2000 + Stage2 2000 for
both fresh control and outer2 candidate, followed by paired DAVIS50 evaluation.

Evaluation tooling added for the corrected pair:

- `exp23_two_stage_pool_morphology_sweep/code/export_accelerate_checkpoint_to_diffueraser.py`
- `exp23_two_stage_pool_morphology_sweep/code/summarize_exp23_pair_eval.py`
- `exp23_two_stage_pool_morphology_sweep/scripts/eval_exp23_pair001_davis50_pai.sh`

These tools export intermediate accelerate checkpoints into evaluator-readable
DiffuEraser roots, build Stage1+DPO / SFT-S2 hybrids through the canonical
hybrid builder, and summarize paired DAVIS50 metrics. They do not modify
`inference/metrics.py`.
