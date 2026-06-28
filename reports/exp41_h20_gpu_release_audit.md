# Exp41 H20 GPU Release Audit

Status: `H20_GPU0_7_READY`

Host: `instance-afs92r3e`

## Initial State

At `2026-06-29T05:27:14+08:00`, GPU4 had one non-system compute process:

- GPU: `4`
- PID: `3365990`
- PPID: `3365988`
- PGID: `3365988`
- SID: `3365988`
- user: `ubuntu`
- elapsed: about `38:33`
- VRAM: `51514 MiB`
- utilization: `100%`
- cwd:
  `/home/nvme04/workspace/world_model_phys/PHYS/world_model_phys_stageA_v5_broad_lora_work`
- exe:
  `/home/nvme03/workspace/lingbot-world/.conda_envs/lingbot-world-v2/bin/python3.10`
- command:
  `python -m cam_physgeo.dpo.failure_diagnostics sigma-sensitivity ... --device cuda --runtime_device cuda`

This task was unrelated to Video_inpainting_DPO, MiniMax, or Exp41 and was not a
system process.

## Cleanup

Action:

```text
kill -TERM -- -3365988
```

After 30 seconds the process group was gone. No `KILL` was required. No
`pkill python`, `killall python`, or GPU reset was used.

## Final State

At `2026-06-29T05:28:10+08:00`:

| GPU | memory used | utilization | compute app |
| --- | ---: | ---: | --- |
| 0 | 28 MiB | 0% | none |
| 1 | 1 MiB | 0% | none |
| 2 | 1 MiB | 0% | none |
| 3 | 1 MiB | 0% | none |
| 4 | 1 MiB | 0% | none |
| 5 | 1 MiB | 0% | none |
| 6 | 1 MiB | 0% | none |
| 7 | 1 MiB | 0% | none |

`nvidia-smi --query-compute-apps` returned no compute apps. GPU0 still has the
normal Xorg graphics process, and `nvitop` holds `/dev/nvidia*` file handles,
but neither is a compute process.

Decision:

```text
H20_GPU0_7_READY
```
