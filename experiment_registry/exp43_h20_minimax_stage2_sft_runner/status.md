# Exp43 Status

Current status: `H20_EXP43_BF16_SAFE_READY`

## 2026-06-29 BF16 Safe Preflight

Current status: `H20_EXP43_BF16_SAFE_READY`

- Added Exp43-isolated BF16-safe runner, precision policy, launchers, config,
  preflight manifest copy, and unit tests.
- P0-P7 passed on H20.
- P7 used DDP8 over GPU0-GPU7.
- Rank0 checkpoint save/reload passed for P4, P5, P6, and P7.
- Peak rank0 memory:
  - P4 fp32 one-batch train: `68982.773` MiB.
  - P7 DDP8 bf16 one-batch train: `62087.76` MiB.
- Warnings observed: `expandable_segments` unsupported on this H20 platform and
  DDP `find_unused_parameters=True` found no unused parameters. These did not
  block finite loss/gradients or checkpoint reload.
- No SIGFPE, OOM, CUDA error, NaN/Inf, or Xid was observed.
- Final GPU0-GPU7 compute apps: none.
- Quality claim: none. This is runtime stability only.

Reports:

- `reports/exp43_h20_bf16_safe_preflight.md`
- `reports/exp43_h20_bf16_safe_preflight.csv`
- `reports/exp43_h20_bf16_safe_preflight_summary.json`

## 2026-06-29 Stage2 SFT Runner Readback

Current status: `H20_EXP43_STAGE2_SFT_RUNNER_READBACK_COMPLETED`

- Start HEAD: `03ce2eb5fdc476789280eaea97f2145a0aa369b5`.
- Exp41 blocker: existing MiniMax SFT/DPO runners cap at 10 steps or less.
- Newly authorized code scope: `exp43_h20_minimax_stage2_sft_runner/` only.
- H20 data readback: Exp41 `H20_MINIMAX_DATA_READY`, `2242` active refs checked,
  `0` missing, Exp40 H20-safe `train64/search24/shadow24` available.
- H20 runtime readback: Torch `2.5.1+cu124`, CUDA `12.4`, 8 GPUs, BF16
  supported.
- Exp41 BF16 readback: P0-P7 passed, including DDP8 one-batch train; no SIGFPE,
  OOM, CUDA error, NaN/Inf, or Xid.
- Exp41 protocol readback: executable official protocol matches current runner
  settings; raw output is primary, no hidden comp, no GT leakage, no mask
  reversal.
- Exp42 readback: PAI-side data mining only; no H20 pseudo-success pool
  available yet.
- VOR-Eval remains excluded.

Reports:

- `reports/exp43_h20_stage2_sft_runner_readback.md`

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
