# PRD 24: DPO Adapter Baseline Feasibility

Date: 2026-06-15

## Scope

This is a feasibility audit only. No adapter training has been launched.

Rules:

- No train code means no DPO adapter training.
- Closed-source or inference-only systems are frozen baselines / related work.
- Non-diffusion models are not direct Diff-DPO adapters.
- Any future adapter code must be isolated in its own experiment folder and
  registry.

## Feasibility Classes

### A. Direct Diffusion Adapter Candidates

- `VideoPainter`
- `FFF-VDI`

These have public training entrypoints and are diffusion-style enough to justify
a future smoke-test gate, but compute risk is high.

### B. Output-Level Preference Candidates

- `ProPainter`
- `E2FGVI`
- `STTN`

These have training code, but they are not diffusion noise-prediction models.
They should not be described as direct Diff-DPO adapters.

### C / D. Frozen Baseline Or Related Work Only

- `FloED`
- `CoCoCo`
- `VACE`
- `MiniMax-Remover`
- `RT-Remover`
- `LGVI`
- `VideoComp`

No adapter-ready training path was validated for these in the current audit.

## Recommendation

Do not launch training yet. Finish:

1. YouTubeVOS100 + DAVIS50 extended eval.
2. Final 20 paper/PPT visual cases.
3. Then consider a `VideoPainter` adapter gate only with explicit confirmation.

Reports:

```text
reports/dpo_adapter_baseline_feasibility_table.md
reports/dpo_adapter_baseline_feasibility_table.csv
reports/adapter_gate_candidate_recommendation.md
```

