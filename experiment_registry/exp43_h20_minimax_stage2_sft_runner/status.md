# Exp43 Status

Current status: `H20_EXP43_GPU_RELEASE_AUDITED`

## 2026-06-29 H20 GPU Release Audit

- Host: `instance-afs92r3e`.
- H20 GPU0-GPU7 compute apps: none.
- GPU memory/utilization:
  - GPU0: `28 / 97871 MiB`, `0%`, Xorg graphics only.
  - GPU1-GPU7: `1 / 97871 MiB`, `0%`.
- `nvitop` holds `/dev/nvidia*` file handles but is not a compute process.
- CUDA smoke passed in `/home/nvme01/miniconda3/envs/wan`.
- Torch: `2.5.1+cu124`; CUDA: `12.4`; BF16 supported: true.
- Killed PID/PGID: none.
- PAI actions: none.

Reports:

- `reports/exp43_h20_gpu_release_audit.md`
- `reports/exp43_h20_gpu_release_audit.csv`
