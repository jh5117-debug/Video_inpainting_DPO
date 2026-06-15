# Status

Status: blocked before smoke.

Reason:

- VideoPainter repo and training entry pass audit.
- Direct Diff-DPO is feasible by design.
- Adapter trainer has not been implemented.
- HAL is not PAI, so no smoke was run.

Next action:

```text
Implement isolated adapter trainer under exp14_adapter_videopainter/code/,
then run Smoke1 on PAI.
```

