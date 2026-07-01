# Exp53 H20 R1/R2 Readback And GPU Audit

Status: `EXP53_H20_PARTIAL_GPU_READY`

Timestamp: `2026-07-01T08:53:57+00:00`
Branch: `research/exp53-void-r1r2-targeted-h20-20260701`
Base: `origin/research/exp52-void-winner-preserving-allgpu-20260701`

## Readback

Exp50/Exp51/Exp52 established that VOID official inference works, same-model loser generation is available, preference forward and zero-gap passed, and vanilla 10-step LoVI-DPO was negative because the margin was loser-dominant. Exp52 reduced loser dominance in R1 row0 smoke and produced mixed R1_Q0 evidence. Exp53 targets Q1/Q2 and T300/T500 for R1/R2 only.

## Current H20 GPU0-3 Audit

| GPU | Used MiB | Total MiB | Util % | Status |
| --- | ---: | ---: | ---: | --- |
| 0 | 9911 | 97871 | 0 | `occupied_or_unknown` |
| 1 | 1 | 97871 | 0 | `free` |
| 2 | 1 | 97871 | 0 | `free` |
| 3 | 9884 | 97871 | 0 | `occupied_or_unknown` |

Compute apps snapshot:

```text
GPU-53e27608-e06c-4088-85fd-81412f1f451d, 2870269, /home/nvme03/SZQ-WAM/miniconda3/envs/fastwam/bin/python, 9874
GPU-0b7c9457-c09b-532c-07aa-fe3ee306411d, 2944336, /home/nvme03/SZQ-WAM/miniconda3/envs/fastwam/bin/python, 9874
GPU-18eab895-41e6-3062-a5df-b104db5e2cd0, 2945149, /home/nvme03/SZQ-WAM/miniconda3/envs/fastwam/bin/python, 9874
GPU-c15fc8f0-9d89-86bb-701d-38f848f9366e, 2949403, /home/nvme03/SZQ-WAM/miniconda3/envs/fastwam/bin/python, 9874
GPU-ca37d462-7bfc-213b-e46e-bef520750458, 491274, /home/nvme03/workspace/lingbot-world/.conda_envs/lingbot-world-v2/bin/python, 90638
```

No unknown process was terminated. No GPU reset, pkill python, or killall python was used.

## Answers

1. Exp50 vanilla 10-step failed because the preference margin grew mainly by degrading the loser branch while heldout video metrics did not improve.
2. Exp51 confirmed loser-dominant behavior and quadmask-local object/boundary damage.
3. Exp52 rescue grid was limited by slow forward and only reached mixed R1_Q0 evidence, though cache and row0 smoke proved the path can run.
4. Current available GPUs for Exp53 are GPU0-3 status: `EXP53_H20_PARTIAL_GPU_READY`.
5. Exp53 differs by targeting R1/R2 on Q1/Q2 and T300/T500, using cached inputs and one-step only.
6. No long training is allowed; 10-step waits for Exp55 cross-server aggregation.
