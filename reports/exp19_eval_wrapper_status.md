# Exp19 Eval Wrapper Status

```text
BLOCKED_PENDING_EXP19_INFERENCE_WRAPPER
```

The existing `tools/run_davis50_framewise_protocol_eval.py` can load
standard DiffuEraser `last_weights`, but it cannot load external
flow encoder / adapter weights or pass flow tensors into the UNet.

Do not evaluate Exp19 by silently falling back to Exp11 weights.
