# Exp50 VOID Zero-Gap Gate

Status: `VOID_TRAINABLE_FORWARD_BLOCKED`

## What Was Verified

- G0 micro data exists: `manifests/exp50_void_adapter_train4.jsonl` and `manifests/exp50_void_adapter_heldout4.jsonl`.
- Runtime official training view created at `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp50_pai_void_adapter_feasibility/g1_official_train_view`.
- Official dataset view adds `mask.mp4` symlinks to Gate8 `quadmask_0.mp4`; original Gate8 files and official source were not modified.
- Official `ImageVideoDataset` bucket mode loaded train4 successfully with aligned `pixel_values`, `input_condition`, and `mask` arrays.
- VOR-Eval was not used.

## Why Zero-Gap Did Not Run

The requested G1 gate requires a policy/reference setup with same noise/timestep and finite winner/loser losses. Official VOID `train.py` is an SFT-style MSE trainer over target noise/velocity and does not expose a same-source winner/loser preference forward or frozen reference policy path. The official shell runners also use `--use_deepspeed`; deepspeed remains intentionally uninstalled because the relay found it can pull the wrong torch/CUDA stack.

Running a single optimizer step through the official SFT path would not satisfy the requested G1 preference/zero-gap requirements. Therefore zero-gap is blocked before optimizer or backward.

## Safety

- Training run: no.
- Optimizer step: no.
- Backward: no.
- 10-step: not run.
- VOID official source modified: no.
- PAI base env modified: no.

Exact blocker: `VOID_TRAINABLE_FORWARD_BLOCKED_PREFERENCE_WRAPPER_REQUIRED`.
