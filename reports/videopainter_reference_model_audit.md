# VideoPainter Reference Model Audit

Status: feasible in principle, not measured.

## Reference Options

Option A:

```text
Load VideoPainter pretrained branch/transformer twice.
Policy = trainable copy.
Reference = frozen copy.
```

Option B:

```text
Load the same checkpoint twice:
policy trainable
reference frozen
```

Option C:

```text
If memory is too high, use no_grad reference forward with CPU/offload or split
policy/reference passes.
```

## Current Paths

Expected upstream paths:

```text
../ckpt/CogVideoX-5b-I2V
../ckpt/VideoPainter/checkpoints/branch
../ckpt/VideoPainterID/checkpoints
```

HAL local cache:

```text
/home/hj/.cache/huggingface/hub/models--TencentARC--VideoPainter
```

## Required Smoke Checks

- policy checkpoint path exists
- reference checkpoint path exists
- policy and reference start from same weights
- reference parameters have `requires_grad=False`
- reference forward uses `torch.no_grad()`
- no reference gradient after backward
- memory footprint recorded for 1-step smoke
- multi-GPU behavior recorded if using Accelerate

## Decision

Reference model is possible, but not validated on PAI. Do not start DPO smoke
until this is implemented and measured.

