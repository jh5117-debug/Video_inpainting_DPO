# Exp14 Gate2000 Precheck

Status: blocked.

The user requested direct gate2000 and no smoke. The smoke guard has been
removed from the launch script, but the hard requirement remains: the isolated
VideoPainter DPO adapter trainer must exist before launching.

Current blocker:

```text
exp14_adapter_videopainter/code/train_videopainter_dpo_adapter.py missing
```

The gate was not launched because upstream VideoPainter training does not
compute policy/reference winner/loser DPO losses and would not be the requested
adapter experiment.

See the global report:

```text
reports/videopainter_adapter_gate2000_precheck.md
```

