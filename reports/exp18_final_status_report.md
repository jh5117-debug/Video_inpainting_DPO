# Exp18 Final Status Report

Date: 2026-06-17

## What Was Completed

- Created `exp18_multiframe_propagation_gated_dpo/`.
- Created `experiment_registry/exp18_multiframe_propagation_gated_dpo/`.
- Added PRD: `PRD/39_exp18_multiframe_propagation_gated_dpo.md`.
- Implemented non-oracle multi-frame propagation cache:
  `exp18_multiframe_propagation_gated_dpo/code/precompute_multiframe_propagation_cache.py`.
- Implemented propagation-aware dataset/loss/trainer scaffolding:
  - `exp18_dataset.py`
  - `exp18_loss.py`
  - `train_exp18_stage1.py`
  - `exp18_dpo_diag.py`
- Added PAI scripts:
  - `prepare_exp18_cache_limit100_pai.sh`
  - `launch_exp18_stage1_gates_pai.sh`
  - `eval_exp18_davis10_pai.sh`
  - `make_exp18_visuals_pai.sh`
  - `launch_exp18_overnight_pai.sh`

## What Was Verified Locally

```text
python -m py_compile exp18_multiframe_propagation_gated_dpo/code/*.py
bash -n exp18_multiframe_propagation_gated_dpo/scripts/*.sh
synthetic one-sample propagation cache dry-run
```

All passed.

## What Was Not Run

No real PAI cache, training, or eval was launched from this HAL session.

Reason:

```text
ssh pai: hostname cannot be resolved
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO not visible
/mnt/workspace/hj/nas_hj/data/external/davis_432_240 not visible
```

## Current Conclusion

Exp18 is implementation-ready but not experimentally evaluated. Current best
remains:

```text
Exp11 outer b0.75 S2
```

## Required Next Step

Run from PAI:

```bash
bash exp18_multiframe_propagation_gated_dpo/scripts/launch_exp18_overnight_pai.sh
```

The launcher will stop if the generated-loser manifest, SFT-48000 weights,
base model, VAE, or DAVIS path are missing.

