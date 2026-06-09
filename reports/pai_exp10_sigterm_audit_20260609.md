# PAI Exp10 SIGTERM Audit - 2026-06-09 CST

## Summary

PAI Exp10 is not currently running. The recent failures are repeated external
`SIGTERM` events, not CUDA OOM, host OOM, `SIGFPE`, or an ordinary training-code
exception.

## Current Verified State

- Exp9 completed Stage1, Stage2, Stage1 DAVIS validation, and Stage2 DAVIS
  validation under `RUN_VERSION=20260609_025331_d3n16_val24`.
- Exp10 original PAI Stage1 reached step 1350. Complete checkpoints exist at
  `checkpoint-500` and `checkpoint-1000`.
- `checkpoint-1000` was exported to policy-init format as:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260609_025331_d3n16_val24_exp10_region_local_dpo_s1_2000_davis_pai/checkpoint-1000_policy_init
```

- Exp10 continuation attempts from that policy-init were externally terminated.
- Exp11 remains blocked by implementation audit; do not launch it until
  train-time flow/prior consistency is safely implemented and audited.

## Evidence

Foreground SSH continuation run:

```text
logs/pipelines/exp10_region_local_dpo_s1s2_2000_davis_pai_policyinit1000_sshfg_gpus0_3_20260609_131212.log
```

Result: reached about step 40, then:

```text
traceback : Signal 15 (SIGTERM) received
```

Named Python executable run:

```text
logs/pipelines/exp10_region_local_dpo_s1s2_2000_davis_pai_20260609_1328_d3n16_val24_exp10_namedpy_policyinit1000_gpus0_3_20260609_132805.log
```

Result: reached `checkpoint-25` save, wrote model weights, then received
`SIGTERM` before the checkpoint completed.

`lingbotworld-phy` executable run:

```text
logs/pipelines/exp10_region_local_dpo_s1s2_2000_davis_pai_20260609_1336_d3n16_val24_exp10_policyinit1000_lingbotworldphy_gpus0_3_20260609_133650.log
```

The stage log confirmed the actual worker Python executable:

```text
[dpo-stage1] python_runner=/mnt/nas/hj/conda_envs/diffueraser/bin/lingbotworld-phy
[dpo-stage1] accelerate_python=/mnt/nas/hj/conda_envs/diffueraser/bin/lingbotworld-phy
```

Result: reached about step 6, then:

```text
traceback : Signal 15 (SIGTERM) received by PID 2672520
```

## Interpretation

The same termination repeats under:

- foreground SSH, without relying on terminal detachment;
- named Python worker executable `lingbot-worldmodel`;
- named Python worker executable `lingbotworld-phy`.

This rules out the simple explanations that the run is killed because of
`nohup`, terminal close, generic `python` worker names, OOM, or a Python
exception.

## Administrator Request

Ask the PAI/DSW administrator to inspect node or platform logs around:

```text
2026-06-09 13:16:58 CST: workers around 2665045-2665048
2026-06-09 13:31:49 CST: workers around 2669008-2669011
2026-06-09 13:38:50 CST: workers around 2672520-2672523
2026-06-09 14:55:32 CST: fresh no-resume worker PID 2698917
```

Required outcome before relaunching PAI Exp10/Exp11:

- identify the sender/policy issuing `SIGTERM`; or
- allowlist the job/process; or
- provide the correct approved PAI job wrapper/process naming convention.

Until then, relaunching on PAI is expected to waste time and create incomplete
checkpoints.

## Fresh No-Resume Retry Evidence

A fresh Exp10 run was launched on PAI to test whether the previous failures were
caused by a corrupted/interrupted resume checkpoint.

```text
RUN_VERSION=20260609_145145_exp10_fresh_d3n16_val24
log=logs/pipelines/exp10_region_local_dpo_s1s2_2000_davis_pai_20260609_145145_exp10_fresh_d3n16_val24_fresh_gpus0_6_20260609_145145.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
NUM_GPUS=7
RESUME_FROM_CHECKPOINT=none
POLICY_INIT_PATH=
```

Observed failure:

```text
06/09/2026 14:54:41 - Total optimization steps = 2000
06/09/2026 14:54:51 - [dpo_diag] global_step=1 ...
W0609 14:55:32 ... Sending process 2698918 closing signal SIGTERM
traceback : Signal 15 (SIGTERM) received by PID 2698917
training/dpo/lingbot-worldmodel-stage1.py FAILED
```

Conclusion: the fresh run also receives external SIGTERM around step 6. This
rules out the interrupted `checkpoint-1000_policy_init` resume path as the
primary explanation.
