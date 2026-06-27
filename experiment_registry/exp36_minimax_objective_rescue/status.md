# Exp36 Status

Current status: `EXP36_READBACK_COMPLETED`

## 2026-06-27 Readback

- Branch: `research/exp36-minimax-objective-rescue-20260627`.
- Base HEAD: `fb70266d53f5f9abd5e8d09ef9d2de324a10b7d6`.
- Exp30 status read: `MINIMAX_ADAPTER_RECIPE_NOT_READY`.
- Exp35 status read: `MINIMAX_RESCUE_RECIPE_NOT_READY`.
- Failure class entering Exp36: objective/update-scale and hard-state design
  remain suspect; code-load and inference-path failure are not supported by
  Exp35 evidence.
- Protected lanes checked read-only.
- No GPU inference, training, 30-step, RC-FPO, or protected-lane action
  launched by this readback milestone.

Report:

- `reports/exp36_minimax_objective_rescue_readback.md`

## 2026-06-27 No-Change Forensic Audit

- Status: `MINIMAX_NOCHANGE_CAUSE_UTILITY_SCALE_TOO_WEAK`.
- Training launched: false.
- Inference launched: false.
- Checkpoint fallback cause: not supported.
- Trainable-scope cause: not supported by current evidence.
- Dominant cause: Exp30 utility/margins were too weak; Exp35 hard-noise
  rescue produced movement but not useful heldout repair.
- Exp35 rescue visual better rows: `0/48`.

Reports:

- `reports/exp36_minimax_nochange_forensic_audit.md`
- `reports/exp36_minimax_nochange_summary.json`

## 2026-06-27 Inference Sensitivity Test

- Status: `MINIMAX_INFERENCE_SENSITIVITY_PASS`.
- Training launched: false.
- GPU used: PAI GPU0.
- Runtime PID/PGID: `1898322` / `1898322`.
- Identity control max full MAE: `0.0`.
- Perturbed mean full/mask MAE: `0.08821829589193357` /
  `0.15630244233590715`.
- Visual review: `4/4` strips opened.
- Collapse/new artifact count: `0`.
- Quality-positive claim unlocked: false.

Reports:

- `reports/exp36_minimax_inference_sensitivity.md`
- `reports/exp36_minimax_inference_sensitivity_summary.json`
