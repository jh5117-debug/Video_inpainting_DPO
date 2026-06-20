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

