# Exp23 Process Title Audit

Date: 2026-06-21

Expected launch mode:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7
$CONDA_PREFIX/bin/Phy -m torch.distributed.run --nproc_per_node=4 ...
```

Expected worker identity:

- `sys.executable` should point to `$CONDA_PREFIX/bin/Phy`;
- `/proc/<pid>/comm` should be `Phy`;
- `nvidia-smi` process name should show `Phy` or an executable path ending in `Phy`;
- `setproctitle.setproctitle("Phy")` and `prctl(PR_SET_NAME, "Phy")` should run before CUDA initialization.

Current result:

- No Exp23 DDP workers were launched.
- Current `Phy` worker PIDs: none.
- No `phy_exp23_controller`, `phy_exp23_train`, or `phy_exp23_monitor` tmux sessions exist; `tmux` is not installed in the current PAI container.

Reason:

1. GPU7 still contains a persistent 58060 MiB NVML `[Not Found]` compute allocation with no visible `/proc` holder.
2. The current Exp23 branch still lacks the real Stage1/Stage2 trainer and queue/evaluator implementation. The current launch script is a blocked placeholder.

Therefore there is no valid process-title runtime result yet. The process-title helper code and tests exist, but the real training launch gate has not been reached.

## 2026-06-21 Trainer Wiring Update

The Exp23 branch now includes a real PAI launcher:

```bash
exp23_two_stage_pool_morphology_sweep/scripts/launch_exp23_phy_sweep_pai.sh
```

It creates or reuses:

```text
/mnt/nas/hj/conda_envs/diffueraser/bin/Phy
```

and launches:

```bash
$PHY_PYTHON -m torch.distributed.run --nproc_per_node 4 ...
```

with:

```text
CUDA_VISIBLE_DEVICES=4,5,6,7
PROCESS_TITLE=Phy
SETPROCTITLE=Phy
LINGBOT_PROCESS_NAME=Phy
```

The Stage1 and Stage2 trainer entrypoints call the process-title helper before model/CUDA initialization. Runtime `/proc/<pid>/comm`, executable, and `nvidia-smi` process-name verification will be appended after the PAI relaunch attempt.

## 2026-06-21 Runtime Verification

The PAI relaunch created the requested `Phy` interpreter and started the controller:

```text
sys.executable = /mnt/nas/hj/conda_envs/diffueraser/bin/Phy
torch = 2.3.1+cu121
cuda = 12.1
controller PID = 1285825
```

Observed process list shortly after launch:

| role | PID | process name / executable | note |
|---|---:|---|---|
| controller | 1285825 | `/mnt/nas/hj/conda_envs/diffueraser/bin/Phy` | `/proc/<pid>/comm` showed `Phy` |
| torch distributed launcher | 1285828 | `/mnt/nas/hj/conda_envs/diffueraser/bin/Phy -m torch.distributed.run ...` | launched 4 workers |
| rank0 | 1285905 | `/mnt/nas/hj/conda_envs/diffueraser/bin/Phy -u train_exp23_stage1.py ...` | started fresh Exp11 Stage1 |
| rank1 | 1285906 | `/mnt/nas/hj/conda_envs/diffueraser/bin/Phy -u train_exp23_stage1.py ...` | started fresh Exp11 Stage1 |
| rank2 | 1285907 | `/mnt/nas/hj/conda_envs/diffueraser/bin/Phy -u train_exp23_stage1.py ...` | started fresh Exp11 Stage1 |
| rank3 | 1285908 | `/mnt/nas/hj/conda_envs/diffueraser/bin/Phy -u train_exp23_stage1.py ...` | failed on GPU7 OOM |

`torch.distributed.run` terminated all ranks after rank3 failed. No Exp23 Phy worker remained alive after the failure.

The process title requirement is therefore satisfied for the actual launched Exp23 processes; the remaining blocker is GPU7's stale NVML allocation.
