# Exp11 Flow-Prior Truth Audit

Date: 2026-06-11

Scope: local synced repository audit plus local PAI-derived reports. This audit
does not start training and does not regenerate data.

## Verdict

Exp11 is **not** a valid flow-prior consistency DPO result under the current
implementation standard.

Status:

```text
invalid / mislabeled / blocked
```

Reason:

- The shared training loops `training/dpo/train_stage1.py` and
  `training/dpo/train_stage2.py` do not implement train-time `L_flow`,
  `L_prior`, or `L_boundary`.
- The isolated code under `exp11_flow_prior_consistency_dpo/code/` implements a
  proxy, not true ProPainter-prior / optical-flow consistency.
- Its `L_prior` target is the frozen SFT/reference epsilon prediction, not a
  ProPainter prior frame/tensor aligned to predicted clean output.
- Its `L_flow` is adjacent-frame residual consistency on
  `policy_epsilon - ref_epsilon`, not optical-flow warp consistency and not
  flow-confidence weighted.
- Therefore old Exp11 metric rows must be retained only as historical proxy
  numbers, not as method evidence.

## Checklist

| Requirement | Audit Result | Evidence |
|---|---|---|
| Real `L_flow` enters total loss | fail | proxy residual temporal loss only; no flow tensor / warp target |
| Real `L_prior` enters total loss | fail | frozen-ref epsilon target, not ProPainter prior |
| Real `L_boundary` enters total loss | partial/invalid context | boundary proxy exists only in isolated uncommitted proxy code |
| Stage scripts pass lambdas | invalid | separate launcher previously forced proxy training; now blocked |
| dpo_diag has real flow/prior/boundary columns | unverified / invalid | local committed reports do not prove real targets; proxy columns would not satisfy the definition |
| logs show true flow/prior consistency enabled | fail | existing audit says train-time ProPainter prior and flow are unavailable |
| checkpoint identity proves true Exp11 run | fail/unverified | no local checkpoint proof of a true Exp11 implementation |

## Code-State Fix

`exp11_flow_prior_consistency_dpo/scripts/launch_exp11_pai.sh` now writes a
blocked audit and exits before training. It no longer enables proxy training.

The registry status is downgraded to:

```text
invalid_mislabeled_blocked
```

## Required Before Re-enabling Exp11

- Expose train-time prior frames/tensors or a safe predicted-clean/x0 target.
- Implement `L_prior` against that real ProPainter-prior target.
- Expose flow tensors or a differentiable warp target before claiming `L_flow`.
- Record nonzero real-target diagnostics:
  `prior_loss`, `boundary_loss`, `flow_loss`,
  `exp11_prior_available`, `exp11_boundary_available`,
  `exp11_flow_available`.
- Re-run the implementation audit and only then unblock training.
