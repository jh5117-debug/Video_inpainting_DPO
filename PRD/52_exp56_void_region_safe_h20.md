# Exp56-H20 VOID Local Region-Safe Objective Repair

Date: 2026-07-02

Branch: `research/exp56-void-region-safe-h20-20260701`

Base: `origin/research/exp55-void-crosslane-aggregator-20260701`

Scope:

- H20 lane only.
- Run only `R5_Q2_T500_S0` and `R5_HALF_Q2_T500_S0`.
- Q2 / T500 only.
- S0 `proj_out` only.
- One-step only.
- No 10-step, no old R1/R2/R3/R4 recipe, no VOR-Eval, no hard comp.

Motivation:

Exp55 showed that object/mask and outside can improve while overlap / affected / boundary regress. Exp56-H20 tests whether object-only DPO plus stronger transition-region preservation can keep object gains without damaging overlap, affected, or boundary regions.

Milestone A status: `EXP56_H20_GPU_READY`.

GPU0-3 are available on H20 for this lane. GPU0 has only the expected Xorg graphics allocation; GPU1-3 have no compute process. No stale project process was killed.

Milestone C status: `EXP56_H20_R5_ONESTEP_MIXED`.

Results:

- `R5_Q2_T500_S0`: full PSNR +0.013859, object +0.956095, overlap -0.153271, affected -0.084209, boundary -0.047360, outside +0.047483, visual 0 better / 2 tie / 2 worse.
- `R5_HALF_Q2_T500_S0`: full PSNR +0.012121, object +0.667133, overlap -0.139584, affected -0.113662, boundary -0.069402, outside +0.047722, visual 0 better / 2 tie / 2 worse.

Decision: no one-step PASS. 10-step remains locked until a later cross-lane aggregator finds a PASS candidate.
