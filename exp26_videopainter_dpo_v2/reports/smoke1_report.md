# VideoPainter Adapter Smoke1 Report

Status: **not run**.

Reason:

```text
This session is on HAL, not PAI, and the VideoPainter DPO adapter trainer is
not implemented yet.
```

Smoke1 may only run after:

1. `exp14_adapter_videopainter/code/` contains a copied/adapted VideoPainter DPO
   trainer.
2. The trainer can load winner/loser pairs.
3. The trainer can load frozen reference and trainable policy.
4. DPO diagnostics are written.
5. PAI data/model paths are verified.

Current decision: blocked before smoke.

