# VideoPainter DPO Trainer Preflight

Date: 2026-06-15

## Status

Not run.

## Reason

The required trainer does not exist:

```text
exp14_adapter_videopainter/code/train_videopainter_dpo_adapter.py
```

The requested preflight requires:

- load policy VideoPainter;
- load frozen reference VideoPainter;
- load one winner / loser / mask pair;
- compute `m_w`, `m_l`, `m_w_ref`, `m_l_ref`;
- compute normalized-gap DPO loss;
- run one backward pass;
- verify reference has no gradients.

Those checks cannot be run without the isolated trainer and pair dataloader.

## Decision

Do not start `exp14_adapter_videopainter_gate2000`.

