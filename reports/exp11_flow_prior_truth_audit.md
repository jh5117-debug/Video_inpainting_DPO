# Exp11 Flow-Prior Truth Audit

Date: 2026-06-11

Scope: local synced repository audit plus local PAI-derived reports. This audit
does not start training and does not regenerate data.

## Verdict

Exp11 is **not** a valid real optical-flow / ProPainter-prior consistency DPO
result under the current implementation standard. The existing completed run is
usable only as an `Exp11-proxy` result.

Status:

```text
valid_as_proxy_only / real_flow_prior_blocked
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
- Therefore old Exp11 metric rows must be retained only under the label
  `Exp11-proxy: frozen-ref prior + boundary + temporal residual proxy DPO`,
  not as real flow-prior method evidence.

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
blocked audit and exits before new training. The existing PAI proxy run has
complete Stage1/Stage2 weights and diagnostics; it does not need retraining.

The registry status is downgraded to:

```text
proxy_complete_real_flow_prior_blocked
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
