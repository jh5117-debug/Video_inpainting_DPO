# VideoPainter Adapter Gate2000 Training Monitor

Date: 2026-06-16

## Final Status

```text
status = completed_2000_steps
pid = 659269
gpu = 0
```

The VideoPainter adapter gate2000 completed on PAI using the isolated Exp14
trainer. It did not use upstream VideoPainter official training as a substitute.

## Paths

PAI clean worktree:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate
```

Pipeline log:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/logs/pipelines/exp14_adapter_videopainter_gate2000.log
```

Trainer log:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/exp14_adapter_videopainter/runs/gate2000/train.log
```

DPO diagnostics:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/exp14_adapter_videopainter/dpo_diag/dpo_diagnostics.csv
```

## Checkpoints

All expected checkpoints / final weights exist:

```text
checkpoint-500
checkpoint-1000
checkpoint-1500
checkpoint-2000
last_weights
```

`last_weights` contains:

```text
last_weights/branch/config.json
last_weights/branch/diffusion_pytorch_model.safetensors
last_weights/run_manifest.json
```

## Runtime Health

Observed:

- process exited after completing step 2000;
- GPU0 was released;
- no `Traceback`, `OOM`, `OutOfMemory`, `NaN`, or `BLOCKED` in the monitored logs;
- dpo_diag completed through step 2000.

## DPO Diagnostic Labels

```text
DPO_SATURATED
LOSER_DOMINANT
GRAD_SPIKE_OBSERVED
```

The training gate succeeded technically, but the diagnostics are not clean
enough to claim adapter improvement without DAVIS metric / visual evidence.

## Eval Status

```text
status = blocked_pending_exp14_thin_eval_adapter
```

The upstream VideoPainter eval path is not the project fixed raw6 hard-comp
metric protocol and currently needs compatibility work. No baseline-vs-adapter
DAVIS metric table was produced.
