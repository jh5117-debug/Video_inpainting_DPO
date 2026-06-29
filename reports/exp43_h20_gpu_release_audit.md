# Exp43 H20 GPU Release Audit

Status: `H20_EXP43_GPU_RELEASE_AUDITED`

## Audit

- Host: `instance-afs92r3e`.
- Audit time: `2026-06-29T09:25:56+08:00`.
- Branch start HEAD: `9ffcb19d825668335d71a4d17d06523ccfea4ab5`.
- H20 conda env: `/home/nvme01/miniconda3/envs/wan`.

## GPU State

| GPU | name | memory used MiB | memory total MiB | utilization |
| ---: | --- | ---: | ---: | ---: |
| 0 | NVIDIA H20 | 28 | 97871 | 0% |
| 1 | NVIDIA H20 | 1 | 97871 | 0% |
| 2 | NVIDIA H20 | 1 | 97871 | 0% |
| 3 | NVIDIA H20 | 1 | 97871 | 0% |
| 4 | NVIDIA H20 | 1 | 97871 | 0% |
| 5 | NVIDIA H20 | 1 | 97871 | 0% |
| 6 | NVIDIA H20 | 1 | 97871 | 0% |
| 7 | NVIDIA H20 | 1 | 97871 | 0% |

`nvidia-smi --query-compute-apps` returned only the CSV header. No compute PID
was present.

`nvidia-smi pmon` reported Xorg on GPU0 as graphics type only. `fuser` and
`lsof` showed `nvitop` holding `/dev/nvidia*` handles; this was not a compute
process and was not killed.

## CUDA Smoke

```text
torch 2.5.1+cu124
cuda 12.4
available True
count 8
bf16 True
sum 4096.0
```

## Actions

- Killed PIDs/PGIDs: none.
- GPU reset: no.
- `pkill python`: no.
- `killall python`: no.
- PAI GPU/file/signal actions: none.

## Decision

H20 GPU0-GPU7 are available for Exp43. Next gate is readback and isolated
runner implementation.
