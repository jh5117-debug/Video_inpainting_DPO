# Exp53 H20 VOID R1/R2 Targeted One-Step Rescue Readback + GPU Audit

Status: `EXP53_GPU_BLOCKED`

Branch: `research/exp53-void-r1r2-targeted-h20-20260701`
HEAD: `a46edb656948799754789f73860a29bf1a469a0c`
Created: `2026-07-01T15:59:54+08:00`

H20 lane only: R1/R2 on GPU0-3.

## Source-of-truth readback

- Exp52 final status: VOID adapter-engineering candidate, not third-backbone evidence.
- Exp52 R1_Q0_T500_S0 was mixed: full/object/boundary/outside improved, affected/overlap regressed.
- 10-step remains locked until Exp55 aggregator sees a one-step PASS.

## GPU audit

- GPU0: 26671 MiB used, util 0%, `occupied_or_unknown`
- GPU1: 26644 MiB used, util 2%, `occupied_or_unknown`
- GPU2: 26644 MiB used, util 1%, `occupied_or_unknown`
- GPU3: 26644 MiB used, util 0%, `occupied_or_unknown`

## Required command excerpts

```text
git fetch rc=124
TIMEOUT after 60s
Fetching origin

```

## File read status

- `PRD/00_current_status.md`: present
- `PRD/01_experiment_matrix.md`: present
- `PRD/47_exp50_pai_void_adapter_feasibility.md`: present
- `PRD/48_exp51_void_loser_dominant_rescue.md`: present
- `PRD/49_exp52_void_winner_preserving_allgpu.md`: present
- `experiment_registry/exp50_pai_void_adapter_feasibility/status.md`: present
- `experiment_registry/exp51_void_loser_dominant_rescue/status.md`: present
- `experiment_registry/exp52_void_winner_preserving_allgpu/status.md`: present
- `reports/exp51_void_loser_dominant_forensic.md`: present
- `reports/exp51_void_quadmask_metrics.md`: present
- `reports/exp51_void_quadmask_ablation_data.md`: present
- `reports/exp52_cache_summary.json`: present
- `reports/exp52_r1_row0_smoke.md`: present
- `reports/exp52_rescue_onestep.md`: present
- `reports/exp52_rescue_onestep_summary.json`: present
- `reports/exp52_void_rescue_decision.md`: present
- `reports/exp52_void_next_steps.md`: present
