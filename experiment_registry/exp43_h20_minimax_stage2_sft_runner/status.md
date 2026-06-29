# Exp43 Status

Current status: `H20_EXP43_SFT_BLOCKED`

## 2026-06-29 SFT-A 30-Step Ladder Cell

Current status: `H20_EXP43_SFT_BLOCKED`

- Completed run: `SFT-A_lr3em5_step30`.
- Training status: `TRAIN_DONE`.
- World size: `8`.
- Peak rank0 VRAM: `66414.96484375` MiB.
- No SIGFPE, OOM, CUDA error, NaN/Inf, or Xid was observed during training.
- Search24 and shadow24 evaluation completed with `48` metric rows.
- Gate result: `FAIL`.
- Search deltas:
  - full PSNR `-5.833128230661999`
  - mask PSNR `-4.674487775418862`
  - boundary PSNR `-4.700944147600658`
  - outside PSNR `-7.594088453813615`
  - Ewarp `+0.6460841968352801`
- Shadow deltas:
  - full PSNR `-6.55060498000691`
  - mask PSNR `-4.223185495799285`
  - boundary PSNR `-5.373455771430662`
  - outside PSNR `-8.45318893655187`
  - Ewarp `+0.5934015673112469`
- LPIPS status: `LPIPS_RUNTIME_BLOCKED`.
- Longer SFT, DPO-after-SFT, and 500-step confirmation: not unlocked.
- Final GPU state for Exp43: no Exp43 compute process remained.
- External GPU7 process:
  `/home/nvme03/workspace/lingbot-world/.conda_envs/lingbot-world-v2/bin/python`;
  not touched.

Reports:

- `reports/exp43_h20_stage2_sft_ladder_runtime_blocker.md`
- `reports/exp43_h20_stage2_sft_ladder_runtime_blocker.csv`
- `reports/exp43_h20_stage2_sft_ladder_runtime_blocker.json`

## 2026-06-29 Data Readiness

Current status: `H20_EXP43_DATA_READY`

- Built Exp43 Stage2 SFT manifests:
  - `train64`
  - `search24`
  - `shadow24`
- Full `train96/search32/shadow32` target: not available.
- Scene overlap across train/search/shadow: `0`.
- Required path failures: `0`.
- Optional path failures: `0`.
- VOR-Eval rows: `0`.
- Hard-comp rows: `0`.
- First-frame decode passed for all required condition/winner/mask/loser frame
  directories.
- Source balance:
  - Train BLENDER/REAL: `32/32`.
  - Search BLENDER/REAL: `12/12`.
  - Shadow BLENDER/REAL: `12/12`.

Reports:

- `reports/exp43_h20_data_readiness.md`
- `reports/exp43_h20_data_readiness.csv`
- `reports/exp43_h20_data_manifest_validation.csv`
- `reports/exp43_h20_data_summary.json`

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

## 2026-06-29 Exp44 Pseudo-Success Handoff Path Validation

Current status: `H20_EXP43_EXP44_PSEUDOSUCCESS_PREFLIGHT_BLOCKED_MISSING_TARGETS`

- H20 host: `instance-afs92r3e`
- Exp44 pseudo-success rows train/search/shadow: `24` / `8` / `8`
- Condition paths in H20 mirror: `24` / `8` / `8`
- Mask paths in H20 mirror: `24` / `8` / `8`
- Pseudo-success target frames in H20 mirror: `0` / `0` / `0`
- Pseudo-success target mp4s in H20 mirror: `0` / `0` / `0`
- Training started: `false`
- Optimizer step: `false`
- GT-only SFT started: `false`

The requested pseudo-success SFT 30-step preflight is blocked until the Exp44
targeted mining outputs are mirrored to H20. GT-only SFT remains explicitly
forbidden as the first experiment.

Reports:

- `reports/exp43_exp44_pseudosuccess_path_validation.md`
- `reports/exp43_exp44_pseudosuccess_path_validation.csv`
- `reports/exp43_exp44_pseudosuccess_path_validation.json`

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
