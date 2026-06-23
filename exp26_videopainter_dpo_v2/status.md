# Exp14 Status

Status: **completed training; DAVIS eval blocked pending Exp14 thin eval adapter**.

Run:

```text
experiment = exp14_adapter_videopainter_gate2000
adapter_type = direct_diff_dpo_isolated_trainer
gpu = 0
```

What passed:

- HAL downloaded `TencentARC/VideoPainter`.
- HAL downloaded `THUDM/CogVideoX-5b-I2V`.
- The weights were transferred to PAI with resumable rsync.
- PAI validated the VideoPainter branch checkpoint and CogVideoX base model.
- The isolated trainer `py_compile` passed.
- The gate launcher `bash -n` passed.
- Trainer preflight passed on PAI.
- Gate2000 completed 2000 steps.
- `checkpoint-500`, `checkpoint-1000`, `checkpoint-1500`, `checkpoint-2000`, and `last_weights` exist.
- `dpo_diagnostics.csv` completed through step 2000.

DPO diagnostic labels:

```text
DPO_SATURATED
LOSER_DOMINANT
GRAD_SPIKE_OBSERVED
```

DAVIS eval:

```text
status = blocked_pending_exp14_thin_eval_adapter
```

The upstream VideoPainter eval path is not the fixed raw6 hard-comp
`inference/metrics.py` protocol and currently fails without additional
compatibility work. Do not claim VideoPainter adapter quality improvement yet.

Important:

This run did not use upstream VideoPainter official training as a replacement
for DPO. It used the isolated Exp14 trainer that computes policy/reference
winner/loser losses and region-local normalized-gap DPO.
