# Exp36 Status

Current status: `MINIMAX_TRAINABLE_SCOPE_EXPANDED_S1_READY`

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

## 2026-06-27 Trainable Scope Audit

- Status: `MINIMAX_TRAINABLE_SCOPE_EXPANDED_S1_READY`.
- Training launched: false.
- Inference launched: false.
- GPU used: none.
- S0: previous full-transformer MiniMax scope, retained as historical
  baseline.
- S1: LoRA attention q/k/v/out + projection scope ready for winner-SFT
  positive-control.
- S2: S1 + last-four-block MLP LoRA, still locked until S1 shows useful
  positive-control evidence.
- Quality-positive claim unlocked: false.

Reports:

- `reports/exp36_minimax_trainable_scope_audit.md`
- `reports/exp36_minimax_trainable_scope_summary.json`

## 2026-06-27 Winner-SFT Positive-Control

- Status: `MINIMAX_POSITIVE_CONTROL_PASS_HELDOUT_QUALITY_NOT_POSITIVE`.
- Training type: supervised winner-SFT positive-control, not DPO.
- GPU used: PAI GPU0.
- Scopes: S0 full-transformer baseline and S1 LoRA attention/projection.
- Strict reload: passed.
- Visual review: `24/24` strips opened.
- Better rows: `0/24`.
- Tie / no visible gain: `20/24`.
- Clearly worse / new artifact: `4/24`.
- Next milestones locked: bad-noise miner, objective rescue, and 30-step confirmatory.

Reports:

- `reports/exp36_minimax_winner_sft_positive_control.md`
- `reports/exp36_minimax_winner_sft_summary.json`

## 2026-06-27 Paper Positioning and Exp36 Stop Decision

Final MiniMax status: `MINIMAX_PLUMBING_POSITIVE_RECIPE_NOT_READY`.

Paper claim status: `TWO_BACKBONE_PLUS_MINIMAX_PLUMBING_ONLY`.

Exp36 proves MiniMax is not ignored by inference and is technically trainable: checkpoint loading, weight sensitivity, strict reload, and winner-SFT parameter/output movement all work. However, quality-positive evidence is absent. Winner-SFT heldout review found `0/24` visually better rows, while prior Exp30/Exp35 preference recipes also had `0` visual better rows.

Decision: do not run bad-noise mining, objective rescue, 30-step confirmatory, RC-FPO, or long training from this state. MiniMax remains a flow-style candidate requiring objective/data redesign, not third-backbone success.

Reports:

- `reports/exp36_minimax_paper_positioning.md`
- `reports/exp36_minimax_paper_positioning.csv`
