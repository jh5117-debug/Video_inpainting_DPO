# PRD 33: OR Method Runtime Status

Date: 2026-06-16

## Rule

A method is only reported as a result if it has:

1. a verified local/PAI repo or project wrapper;
2. real weights;
3. an OR-compatible video+mask inference entry;
4. raw output frames saved for DAVIS50;
5. OR metrics computed without comp.

Blocked methods remain in the report and visual grid as `BLOCKED` rather than
being silently dropped.

## Runtime Status

| Method | Status | Reason |
|---|---|---|
| ProPainter | COMPLETED_50_50 | Existing project wrapper and ProPainter weights ran successfully on DAVIS50. |
| CoCoCo | BLOCKED_NO_WEIGHT | Repo and COCOCO checkpoints exist, but the required SD inpainting diffusers dependency is incomplete. |
| DiffuEraser SFT-48000 | COMPLETED_50_50 | Exp15 isolated DiffuEraser OR wrapper and SFT-48000 weights ran successfully. |
| Ours Exp11 outer b0.75 S2 | COMPLETED_50_50 | Same Exp15 isolated DiffuEraser OR wrapper, using Exp11 outer b0.75 S2 weights. |
| MiniMax-Remover | BLOCKED_IMPORT_ERROR | Repo/weights exist, but current env is too old; needs isolated env with newer diffusers. |
| VideoPainter | BLOCKED_NO_OR_WRAPPER | VideoPainter BR eval exists, but no verified DAVIS2017 OR wrapper is ready. |
| FloED | BLOCKED_NO_REPO | No verified local/PAI repo+weights+OR wrapper. |
| VACE | BLOCKED_NO_REPO | No verified local/PAI repo+weights+OR wrapper. |
| VideoComposer / VideoComp | BLOCKED_NO_REPO | No verified local/PAI repo+weights+OR wrapper. |

Canonical status files:

```text
reports/exp15_or_method_runtime_status.csv
reports/exp15_or_method_runtime_status.md
experiment_registry/exp15_or_benchmark_davis50/status.md
```

## Notes

- MiniMax should not be run in the shared DiffuEraser env.
- VideoPainter should not be evaluated with the previous BR adapter wrapper and
  mislabeled as OR.
- FloED, VACE, and VideoComposer can be revisited only after repo/weights/OR
  wrappers are verified.
