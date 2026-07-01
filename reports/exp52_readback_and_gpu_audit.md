# Exp52 Readback and All-GPU Audit

Status: `EXP52_ALL_GPU_READY`

## Git

- Base Exp51 HEAD: `c9e0c3de994263743d4734890115fd06a913b9df`
- Branch: `research/exp52-void-winner-preserving-allgpu-20260701`
- H20 worktree: `/home/nvme01/H20_Video_inpainting_DPO_exp52_void_allgpu_rescue`

## Required Answers

1. What exactly failed in Exp50 vanilla 10-step?
   - Vanilla LoVI-DPO completed technically but was quality-negative: full PSNR -0.000965, mask PSNR -0.229878, boundary PSNR -0.063034, visual 0 better / 3 tie / 1 worse. Diagnostics showed the DPO margin grew mainly through loser degradation rather than winner improvement.
2. What did Exp51 confirm?
   - `VOID_LOSER_DOMINANT_CONFIRMED`; quadmask-aware metrics highlighted local object/affected/boundary damage; Q1/Q2 quadmask variants are safer than broad Q3; native Kubric data is blocked by missing Kubric/PyBullet/Blender/HUMOTO assets.
3. Why rescue grid was blocked?
   - `VOID_RESCUE_ONESTEP_BLOCKED_SLOW_FORWARD_NO_CHECKPOINT`: R1-R4 grid did not create a checkpoint/report inside the bounded micro window and was terminated.
4. Which GPUs are available?
   - H20 GPU0-7 are free. `nvidia-smi` compute apps were empty; GPU0 used about 28 MiB, GPU1-7 about 1 MiB.
5. What will be different in Exp52?
   - Exp52 first profiles and caches one-row/train4 inputs, validates cache parity, then runs R1 row0 before any full grid. It uses all GPUs only after preregistration and after slow-forward is solved.
6. Why no long training yet?
   - VOID is not third-adapter evidence; vanilla 10-step was negative and Exp51 rescue did not reach checkpoint. Exp52 must pass cache, row0 smoke, one-step video/metric gates before any 10-step micro validation.

## GPU Action

No process was killed. No stale Exp50/Exp51/Exp52 GPU process was found.

## Snapshot

Raw GPU/process snapshot is stored at:

`/home/nvme01/H20_Video_inpainting_DPO/runtime/exp52_void_winner_preserving_allgpu/milestone_a_gpu_process_snapshot.txt`
