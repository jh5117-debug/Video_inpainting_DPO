# Exp31 VideoPainter Resume Policy Audit

Date: 2026-06-27

Status: `VIDEOPAINTER_2000_FRESH_FROM_STEP0`

## Audited Source

- source run: `vp_primary32_50step_20260625_171032`
- source checkpoint:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp26_videopainter_dpo_v2/vp_primary32_50step_20260625_171032/checkpoint-50`
- trainer state:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp26_videopainter_dpo_v2/vp_primary32_50step_20260625_171032/checkpoint-50/trainer_state.pt`

## Read-Only PAI Audit

The Step50 checkpoint tree contains:

- `checkpoint-50/branch/config.json`
- `checkpoint-50/branch/diffusion_pytorch_model.safetensors`
- `checkpoint-50/trainer_state.pt`

`trainer_state.pt` contains:

| field | value |
| --- | --- |
| keys | `optimizer`, `step` |
| step | `50` |
| optimizer state | present |
| optimizer state tensors | `60` |
| optimizer param groups | `1` |
| scheduler state | absent |
| RNG state | absent |

`run_config.json` exists and records:

| field | value |
| --- | --- |
| max_train_steps | `50` |
| lr_scheduler | `constant` |
| seed | `20260625` |

## Decision

The prompt requires full optimizer/scheduler state to label a run as a
continuation from Step50. Step50 has optimizer state but does not have scheduler
state or RNG state. Therefore Exp31 must not be labeled as a true continuation.

Decision:

```text
VIDEOPAINTER_2000_FRESH_FROM_STEP0
```

The 2000-step run will start fresh from the official VideoPainter branch
checkpoint using the same primary32 manifest, policy initialization, reference,
optimizer, scheduler, seed, data order, noise/timestep schedule, effective
batch, first-frame semantics, and formal 49-frame protocol as Exp26.

Forbidden label:

```text
VIDEOPAINTER_2000_CONTINUED_FROM_STEP50
```

## Right-Side Protection

The audit was read-only. No Exp26, Exp30, MiniMax, lock, heartbeat, PID, or
output file was modified. No signal was sent.

