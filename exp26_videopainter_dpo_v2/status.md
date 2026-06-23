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

## 2026-06-23 L0-L4 Gate

- status: L0_L4_PASSED
- L0 formal 49F native strict-load/inference: passed
- L1 same-batch/noise/timestep parity: passed
- L2 policy=reference zero-gap DPO log(2): passed
- L3 one-step native optimizer + strict save/reload + inference: passed
- L4 10-step smoke: passed, PLUMBING_ONLY_13F
- No long training started.
## 2026-06-23 49F Source Split

- L0-L4 remain passed.
- Formal 49-frame source split construction was attempted.
- The available YouTube-VOS directory contains sparse frame sequences; sampled videos had 36 and 20 frames.
- No formal Gate64 self-loser generation was launched.
- Status: `BLOCKED_INSUFFICIENT_49F_SOURCE`.

## 2026-06-23 VOR-BG Formal Tooling

- Added strict 49F source materializer for the locked VOR-BG train/search/shadow manifests.
- Added deterministic moving BR mask generator with `first_frame_gt` enforced by `mask frame0 = 0`.
- Added locked sampler config recording first-49 unique-frame extraction until a native trainer audit proves a different sampler.
- Gate64 remains pending; no VideoPainter self-loser generation or GPU training was started.

Status: `FORMAL_49F_TOOLING_READY_PENDING_PAI_EXTRACTION`.

## 2026-06-23 Probe4

- Selective VOR-BG extraction: `4/4`.
- Formal 49F materialization: `4/4`.
- Moving BR mask generation: `4/4`.
- First-frame GT mask invariant: `first_frame_sum=0` for all four samples.
- Gate64 not started; this was a source/mask probe only.

Status: `FORMAL_49F_PROBE4_PASSED`.

## 2026-06-23 Probe4 Visual Audit

- Probe4 mask visual evidence generated on PAI and reviewed on HAL.
- 4/4 masks are temporally stable and first-frame-GT safe.
- 4/4 masks are synthetic ellipse/circle masks, not semantic object masks.
- 1/4 has edge-touch caution.
- Gate16/Gate64 remain pending official 49F VideoPainter inference.

Status: `PROBE4_MASK_VISUAL_AUDIT_PASSED_PENDING_OFFICIAL_INFERENCE`.
