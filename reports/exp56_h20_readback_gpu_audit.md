# Exp56-H20 Readback And GPU Audit

Status: `EXP56_H20_GPU_READY`

Machine: `instance-afs92r3e`

Time: `2026-07-02T04:10:30+08:00`

Branch: `research/exp56-void-region-safe-h20-20260701`

Base HEAD: `7a85048d57a6e562a1bf768d83394aa1a39f2375`

Readback:

- Exp55 decision: `EXP55_NO_10STEP_MIXED_ONLY`.
- Best Exp55 candidate: `R1_Q2_T500_S0`, mixed-only.
- Failure pattern: object/mask improves and outside is preserved, but overlap / affected / boundary regress.
- Exp56-H20 is restricted to R5 / R5_HALF on Q2/T500, one-step only.

GPU audit:

- GPU0: H20, 28 MiB used, 0% util, Xorg graphics process only.
- GPU1: H20, 1 MiB used, 0% util, no compute process.
- GPU2: H20, 1 MiB used, 0% util, no compute process.
- GPU3: H20, 1 MiB used, 0% util, no compute process.

No stale Exp50/51/52/53/55/56 project process was found or killed. GPU4 has an unrelated `lingbot-world` process and is outside this lane.

Safety:

- Training run: no
- Optimizer step: no
- 10-step: no
- VOR-Eval: no
- Hard comp: no
- VOID official source modified: no
