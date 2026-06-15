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
VideoPainter direct Diff-DPO is structurally possible.
The isolated trainer is now implemented locally.
Gate2000 is still blocked until PAI preflight passes.
```

## Original Blocker

The original precheck blocked because the required isolated adapter trainer was
missing:

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

## Current Implementation

Implemented after the original block:

```text
exp14_adapter_videopainter/code/train_videopainter_dpo_adapter.py
```

It implements:

- a pair dataloader for the current frame-directory manifest;
- policy branch + frozen reference branch;
- shared noise / timestep winner and loser forwards;
- `m_w`, `m_l`, `m_w_ref`, `m_l_ref`;
- region-local outer-boundary MSE;
- log-ratio normalized-gap clipped-loser-gap winner-anchored DPO;
- diagnostics CSV;
- checkpoint and `last_weights` saving;
- `--preflight_only`.

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

## What Still Needs PAI Validation

PAI checked project roots did not contain the Exp14 adapter folder or trainer
at the time of the original precheck. The new trainer must be synced to PAI.

The current DPO manifest is also not in upstream VideoPainter CSV +
`all_masks.npz` format. It uses frame directories:

```text
win_video_path
final_loser_video_path
mask_path
```

The new trainer includes a pair dataloader for this format, but it has not yet
been run against the real PAI VideoPainter weights.

## Current Status

No training was started.
No checkpoint was written.
No gate2000 dpo_diag was written.
No DAVIS eval was run.

Latest PAI attempt on 2026-06-16 CST:

```text
status = blocked_before_preflight
sync_strategy = clean_worktree
clean_repo = /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate
reason = VideoPainter / CogVideoX weights are missing and PAI cannot reach Hugging Face.
```

The dirty priority repo was not modified. A clean Exp14 worktree was created
instead because the priority PAI repo has local tracked changes and untracked
files that block `git pull --ff-only`.

```text
source_commit = 2e187ee
trainer = exp14_adapter_videopainter/code/train_videopainter_dpo_adapter.py
```

What passed:

- Exp14 files synced in the clean worktree.
- `python -m py_compile` passed.
- `bash -n` passed for the gate launcher.
- VideoPainter code repo was rsynced from HAL to PAI.
- YouTube-VOS, DAVIS, and the generated-loser manifest exist.
- Manifest does not contain `/home/nvme01`.
- GPUs are available.

Current blocker:

```text
missing = third_party/VideoPainter/ckpt/CogVideoX-5b-I2V
missing = third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch
```

Equivalent weight paths under `/mnt/nas/hj/weights`, `/mnt/nas/hj/official_repos`,
and HAL `/home/hj/dpo-2-1-exp` were also checked and missing.

HF download was attempted on PAI:

```text
huggingface-cli download TencentARC/VideoPainter
hf download TencentARC/VideoPainter
```

Result:

```text
huggingface-cli = deprecated / exits failure
hf download = httpx.ConnectError: [Errno 101] Network is unreachable
```

No trainer preflight, gate2000 training, dpo_diag, checkpoint, or DAVIS eval was
run.

## Next Required Work

Provide or mount the official VideoPainter / CogVideoX weights on PAI, then rerun:

```text
third_party/VideoPainter/ckpt/CogVideoX-5b-I2V/
third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch/
```

Suggested transfer from a machine with Hugging Face access:

```bash
rsync -az /path/to/CogVideoX-5b-I2V/ \
  root@47.103.26.60:/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter/ckpt/CogVideoX-5b-I2V/

rsync -az /path/to/VideoPainter/ \
  root@47.103.26.60:/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter/ckpt/VideoPainter/
```

Then rerun:

```text
exp14_adapter_videopainter/scripts/launch_videopainter_adapter_gate2000_pai.sh
```

The launcher now runs trainer preflight first. If preflight fails, gate2000 must
remain blocked.

See:

```text
PRD/29_videopainter_dpo_adapter_trainer.md
reports/videopainter_model_structure_audit.md
reports/videopainter_dpo_trainer_preflight.md
```
