# Exp23 GPU4-7 Force Release Audit

Date: 2026-06-21

PAI host: `dsw-753014-dc85766cb-4v2jj`

User authorization: the user explicitly confirmed that existing GPU4-7 jobs belong to the user / collaborators and may be terminated for Exp23. This supersedes the earlier "do not kill colleague jobs" rule for GPU4-7 only.

## Summary

GPU4, GPU5, and GPU6 were successfully released by terminating the high-expert multigen launcher/worker set that occupied those devices.

GPU7 still has a persistent NVML compute allocation:

- PID: `1758887`
- process name: `[Not Found]`
- memory: `58060 MiB`
- `/proc/1758887`: absent
- `lsof`: no visible holder
- `/proc/driver/nvidia/clients`: no visible holder

Per instruction, I did not use `nvidia-smi --gpu-reset` and did not restart the server. Therefore GPU7 is not fully released.

## Processes Terminated

Targeted TERM was sent to:

| role | PID / PGID target | GPU | command / cwd | result |
|---|---:|---:|---|---|
| high-expert launcher | `1246572` | launcher for GPU4-6 workers | `/usr/local/bin/python3.10 /usr/local/bin/torchrun ... train_memory_dense_adapter_base_v0.py ... --base-expert high`; cwd `/mnt/workspace/zhengqi/multiview-map-v0` | exited after TERM |
| high-expert worker | `-1246702` | 4 | `/usr/local/bin/python3.10 -u /mnt/workspace/xiaoqi/multigen/multiview-map-v0-qxq/tools/train_memory_dense_adapter_base_v0.py ...`; cwd `/mnt/workspace/zhengqi/multiview-map-v0` | exited after TERM |
| high-expert worker | `-1246703` | 5 | same as above | exited after TERM |
| high-expert worker | `-1246704` | 6 | same as above | exited after TERM |

No `KILL` was required for these PIDs because the live check after 30 seconds found no remaining `1246572/1246702/1246703/1246704` processes.

I did not kill the low-expert GPU0-3 process group, even though its launchers shared a common ancestor/session with the high-expert job, because the request specifically scoped cleanup to GPU4-7.

## Memory Before / After

| GPU | before used MiB | after used MiB | status |
|---:|---:|---:|---|
| 4 | 141333 | 244 | released |
| 5 | 141045 | 4 | released |
| 6 | 142321 | 292 | released |
| 7 | 58071 | 58071 | not released: NVML ghost allocation |

## GPU7 Probe

`nvidia-smi -q -i 7 -d PIDS,ACCOUNTING` still reports:

```text
Process ID: 1758887
Type: C
Name:
Used GPU Memory: 58060 MiB
```

But inside the current PAI container:

- `/proc/1758887` does not exist;
- `/proc/driver/nvidia/clients` is empty;
- `lsof /dev/nvidia7` returns no holder;
- `nvidia-smi --query-compute-apps` reports the PID as `[Not Found]`.

This is consistent with an orphaned or cross-namespace NVML allocation. Clearing it would require an administrative action or GPU reset, which was explicitly prohibited.

## Exp23 Launch Decision

Exp23 training was not launched because two launch gates are not satisfied:

1. GPU7 is not fully released.
2. The current Exp23 branch still contains only morphology/aggregation/process-title code and tests. It does not yet contain a real Stage1 trainer, Stage2 trainer, paired queue/controller, or DAVIS50 evaluator. The current `launch_exp23_controller_pai.sh` is still a blocked placeholder.

Launching the current script would not start the requested full two-stage sweep.

## Raw Runtime Logs

PAI runtime logs are stored under:

`/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp23_pool_sweep/exp23_two_stage_pool_morphology_sweep/runtime/gpu_release/`

Key files:

- `pre_release_20260621_071137.txt`
- `kill_plan_20260621_071157.csv`
- `kill_detail_20260621_071157.txt`
- `kill_actions_20260621_071229.log`
- `gpu7_ghost_probe_20260621_071324.txt`

