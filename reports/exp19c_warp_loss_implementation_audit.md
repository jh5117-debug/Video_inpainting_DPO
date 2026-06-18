# Exp19c Warp Loss Implementation Audit

Status:

```text
IMPLEMENTED_AND_RAN_ON_PAI
```

Implementation:

- code: `exp19c_light_warp_dpo/code/latent_warp_loss.py`
- trainer: `exp19c_light_warp_dpo/code/train_exp19c_stage2_adapter.py`
- launcher: `exp19c_light_warp_dpo/scripts/launch_exp19c_warp_sweep_pai.sh`

The implementation keeps the Exp11 outer b0.75 S2 DPO loss unchanged and adds a
light confidence-gated latent warp term:

```text
L_total = L_Exp11 + lambda_warp * L_warp
```

What is used:

- predicted clean latent `z_hat0` recovered from the scheduler prediction;
- forward and backward completed flow resized to latent resolution with vector
  scaling;
- forward/backward valid warp masks;
- flow confidence and mask/outer-boundary gate;
- same Exp19b flow-adapter injection modules and wrapper.

What is not used:

- no GT-error flow confidence;
- no flow-warped RGB target;
- no DiffuEraser base unfreezing;
- no Exp11/shared training code modification.

Runs:

| Variant | lambda_warp | Start checkpoint | Status |
| --- | ---: | --- | --- |
| lambda000 | 0.000 | Exp19b-500 | complete |
| lambda005 | 0.005 | Exp19b-500 | complete |
| lambda010 | 0.010 | Exp19b-500 | complete |
| lambda020 | 0.020 | Exp19b-500 | complete |

Diagnostics:

- `base_grad_norm = 0.0` throughout logged rows.
- `adapter_grad_norm > 0`.
- `warp_loss` finite.
- no NaN/OOM/Traceback observed.

Note: during live code sync a pre-existing `lambda000_dpo_diagnostics.csv`
file was accidentally overwritten while training continued. The run completed
and final diagnostics/checkpoints are available; this did not affect training,
but early lambda000 CSV rows before the sync should be treated as log-backed
rather than CSV-backed.
