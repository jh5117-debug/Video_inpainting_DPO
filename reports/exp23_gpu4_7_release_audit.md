# Exp23 GPU4-7 Release Audit

Audit time: 2026-06-21 06:50 CST on PAI host `dsw-753014-dc85766cb-4v2jj`.

Requested Exp23 GPUs: `4,5,6,7`.

Result: `BLOCKED_GPU4_7_NOT_AVAILABLE`.

## GPU State

| GPU | Memory | Util | PID | Assessment | Action |
|---:|---:|---:|---:|---|---|
| 4 | 141333 MiB | 100% | 1246702 | other active project: `/mnt/workspace/xiaoqi/multigen/...`, cwd `/mnt/workspace/zhengqi/multiview-map-v0` | do not kill |
| 5 | 141045 MiB | 100% | 1246703 | other active project: `/mnt/workspace/xiaoqi/multigen/...`, cwd `/mnt/workspace/zhengqi/multiview-map-v0` | do not kill |
| 6 | 142321 MiB | 100% | 1246704 | other active project: `/mnt/workspace/xiaoqi/multigen/...`, cwd `/mnt/workspace/zhengqi/multiview-map-v0` | do not kill |
| 7 | 58071 MiB | 0% | 1758887 reported by NVML, no `/proc/1758887` entry | unknown memory owner | do not reset |

`nvidia-smi pmon` confirmed active compute on GPUs 4-6. `fuser` is not installed in the PAI image; `lsof` returned no additional usable owner for GPU7. `tmux` is not installed.

## Decision

No GPU process was terminated. GPUs 4-6 are not attributable to this project. GPU7 has unknown residual memory without a live proc entry, and the task explicitly forbids GPU reset. Exp23 full training must wait until GPU4-7 are safely available or the user/admin confirms these specific processes can be stopped.

## Current Exp23 State

- HAL branch created: `research/exp23-two-stage-pool-morphology-sweep`
- HAL worktree: `/home/hj/H20_Video_inpainting_DPO_exp23_pool_sweep`
- Exp23 training not launched.
- Process-title/Phy verification not launched because no valid GPU4-7 slot exists.
