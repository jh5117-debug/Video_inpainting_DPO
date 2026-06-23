# VideoPainter Adapter Gate2000 Training Monitor

Date: 2026-06-16

Status: completed_2000_steps.

PAI clean worktree:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate
```

Run:

```text
pid = 659269 (finished)
gpu = 0
max_steps = 2000
checkpointing_steps = 500
```

Logs:

```text
logs/pipelines/exp14_adapter_videopainter_gate2000.log
exp14_adapter_videopainter/runs/gate2000/train.log
```

DPO diagnostics:

```text
exp14_adapter_videopainter/dpo_diag/dpo_diagnostics.csv
```

Latest observed state:

- step reached 2000;
- `checkpoint-500`, `checkpoint-1000`, `checkpoint-1500`, `checkpoint-2000`, and `last_weights` exist;
- diagnostics completed through step 2000 with finite losses;
- no `Traceback`, `OOM`, `OutOfMemory`, `NaN`, or `BLOCKED` observed.

DAVIS eval:

```text
status = blocked_pending_exp14_thin_eval_adapter
```
