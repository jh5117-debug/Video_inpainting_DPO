# VideoPainter DPO Trainer Preflight

Date: 2026-06-16

## Status

```text
status = passed_on_pai
```

The isolated Exp14 trainer preflight passed on PAI after the missing
VideoPainter / CogVideoX weights were downloaded on HAL and transferred to the
PAI clean worktree.

## Weight Resolution

HAL download path:

```text
/home/hj/dpo-2-1-exp/third_party_baselines/VideoPainter/ckpt
```

PAI target path:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter/ckpt
```

Sources:

- `TencentARC/VideoPainter`
- `THUDM/CogVideoX-5b-I2V`

Transfer method:

```text
rsync --partial --append-verify
```

The transferred files passed the PAI required-path checks and were verified not
to be Git LFS pointer stubs.

## PAI Preflight

Clean worktree:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate
```

Preflight report:

```text
exp14_adapter_videopainter/runs/preflight/preflight_report.json
```

Preflight diagnostics:

```text
exp14_adapter_videopainter/dpo_diag/preflight_dpo_diagnostics.csv
```

Key preflight values:

```text
loss = 0.7026171684
dpo_loss = 0.6931471825
m_w = 0.1893996745
m_l = 0.2312944233
m_w_ref = 0.1893996745
m_l_ref = 0.2312944233
grad_norm = 81.0422735139
reference_has_grad = false
```

Interpretation:

- The policy branch and frozen reference branch both loaded.
- Winner / loser forward passes ran with shared noise and timestep.
- The trainer computed `m_w`, `m_l`, `m_w_ref`, and `m_l_ref`.
- Region-local normalized-gap DPO loss was finite.
- Backward succeeded.
- The reference branch stayed frozen.

## Decision

The preflight is sufficient to launch the 2000-step gate. The gate is now
running on PAI using the isolated Exp14 trainer, not upstream VideoPainter
official training.
