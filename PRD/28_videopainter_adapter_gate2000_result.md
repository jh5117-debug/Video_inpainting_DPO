# PRD 28: VideoPainter Adapter Gate2000 Result

Date: 2026-06-15

## Summary

The user requested skipping smoke and launching the VideoPainter adapter
2000-step gate directly. We ran the required minimum precheck first.

Result:

```text
gate2000_precheck_blocked
```

The run was not launched.

After a deeper structure audit, the conclusion is refined:

```text
VideoPainter direct Diff-DPO is structurally possible,
but blocked until an isolated trainer is implemented.
```

## Why Blocked

The required isolated adapter trainer is missing:

```text
exp14_adapter_videopainter/code/train_videopainter_dpo_adapter.py
```

Without this trainer, we cannot compute the requested Exp11 outer b0.75 S2 style
adapter objective:

- policy winner loss `m_w`
- policy loser loss `m_l`
- frozen-reference winner loss `m_w_ref`
- frozen-reference loser loss `m_l_ref`
- region-local MSE with outer boundary weighting
- log-ratio normalized gaps
- clipped loser gap
- winner-anchor terms
- dpo_diag / adapter_diag

Launching the upstream VideoPainter training script would train the official
VideoPainter objective, not a DPO adapter. It would be a mislabeled experiment.

## What Passed

HAL has a local VideoPainter checkout:

```text
/home/hj/dpo-2-1-exp/third_party_baselines/VideoPainter
```

Repo:

```text
https://github.com/TencentARC/VideoPainter
commit bbab6cd5cd5cb89f0e2444305c32fd74a010ae0a
```

Upstream entries exist:

- `train/VideoPainter.sh`
- `train/train_cogvideox_inpainting_i2v_video.py`
- `infer/inpaint.py`
- `evaluate/eval_inpainting.py`

PAI was reachable and has the expected data paths and idle GPUs.

The upstream VideoPainter training loop is diffusion / denoising based and
therefore exposes the kind of latent loss needed for direct DPO in principle.

## What Failed

PAI checked project roots did not contain the Exp14 adapter folder or trainer.
HAL also does not contain the trainer.

The current DPO manifest is also not in upstream VideoPainter CSV +
`all_masks.npz` format. It uses frame directories:

```text
win_video_path
final_loser_video_path
mask_path
```

An isolated pair dataloader is required.

## Current Status

No training was started.
No checkpoint was written.
No dpo_diag was written.
No DAVIS eval was run.

## Next Required Work

Implement the isolated adapter trainer under `exp14_adapter_videopainter/code/`.
Only after that trainer exists should the gate script be rerun.

See:

```text
PRD/29_videopainter_dpo_adapter_trainer.md
reports/videopainter_model_structure_audit.md
reports/videopainter_dpo_trainer_preflight.md
```
