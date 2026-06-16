# Status

Status: completed_training_eval_blocked.

PAI sync strategy: clean_worktree.

Clean repo:
`/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate`

What passed:
- Exp14 trainer and launcher are present.
- Static checks pass.
- VideoPainter code repo is present.
- VideoPainter / CogVideoX weights are present and validated.
- YouTube-VOS, DAVIS, and generated-loser manifest are present.
- Trainer preflight passed.
- Gate2000 launched with the isolated Exp14 trainer.

Current run:
- PID: `659269`
- GPU: `0`
- dpo_diag: `exp14_adapter_videopainter/dpo_diag/dpo_diagnostics.csv`
- run dir: `exp14_adapter_videopainter/runs/gate2000`

Latest observed:
- `checkpoint-500`, `checkpoint-1000`, `checkpoint-1500`, `checkpoint-2000`, and `last_weights` exist.
- `dpo_diagnostics.csv` completed through step 2000.
- No `Traceback`, `OOM`, `OutOfMemory`, `NaN`, or `BLOCKED` observed so far.

DAVIS eval:

```text
status = blocked_pending_exp14_thin_eval_adapter
```

The upstream VideoPainter eval path is not the project fixed raw6 hard-comp
metric protocol and currently needs additional compatibility work. No
baseline-vs-adapter DAVIS metric table exists yet.
