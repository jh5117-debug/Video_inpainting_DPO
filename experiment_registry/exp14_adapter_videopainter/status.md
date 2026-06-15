# Status

Status: isolated trainer implemented; gate2000 blocked until PAI preflight passes.

Reason:

- VideoPainter repo and training entry pass audit.
- Direct Diff-DPO is feasible by design.
- The isolated adapter trainer now exists at
  `exp14_adapter_videopainter/code/train_videopainter_dpo_adapter.py`.
- User requested skipping smoke and going directly to gate2000.
- Gate2000 still must wait for the trainer preflight, because this is the first
  check that policy/reference winner/loser losses and backward pass work on PAI.

Next action:

```text
Sync Exp14 to PAI and run the gate2000 launcher. The launcher now runs
`--preflight_only` first, then starts 2000-step only if preflight succeeds. Do
not launch upstream VideoPainter training as if it were DPO.
```
