# VideoPainter DPO Trainer Preflight

Date: 2026-06-16

Status: passed_on_pai.

The isolated Exp14 VideoPainter DPO trainer preflight passed in the PAI clean
worktree:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate
```

Preflight output:

```text
exp14_adapter_videopainter/runs/preflight/preflight_report.json
```

Preflight diagnostics:

```text
exp14_adapter_videopainter/dpo_diag/preflight_dpo_diagnostics.csv
```

Key values:

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

The pass confirms that the policy/reference branch setup, winner/loser forward
passes, region-local normalized DPO loss, backward pass, and frozen-reference
guard all work on PAI with the real VideoPainter weights.
