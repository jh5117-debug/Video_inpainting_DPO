# Exp25 Pre-Execution Audit

Timestamp: 2026-06-23 CEST/CST

## Scope

This audit starts the next Exp25 phase after the EffectErase VOR core archive
transfer completed. The next phase must not regenerate all 60K VOR losers. It
will build archive/member inventory, selective extraction, canonical OR
manifests, gated OR loser generation, and DiffuEraser OR-DPO data-size scaling.

## HAL

- host: `hal-9000`
- worktree: `/home/hj/H20_Video_inpainting_DPO_exp25_vor`
- branch: `research/exp25-vor-or-preference-data`
- HEAD before this phase: `ca267d0ff2f6edadaedd1c1a0fa03482a8abfc5d`
- status before edits: clean
- `/home/hj` free: about 533G
- GPU state: no Exp25/Exp26 training launched; HAL GPU0 has small unrelated
  python allocations only.

## PAI

- host: `dsw-753014-dc85766cb-4v2jj`
- worktree: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp25_vor`
- branch: `research/exp25-vor-or-preference-data`
- PAI HEAD at audit time: `718fc28ffde63ca746a577b300624eb04b3e4c0d`
- required action: sync PAI to the latest Exp25 branch before running new
  archive or OR jobs.
- NAS free: effectively available under `/mnt/nas`; inode pressure not observed.
- GPU state at audit time:
  - GPU2: idle
  - GPU6: effectively idle
  - GPU0/1/3/4/5: occupied by unrelated `/mnt/workspace/xiaoqi/multigen/...`
    jobs
  - GPU7: still has an unexplained about-58GB allocation and remains excluded

## Exp23 Isolation

No active Exp23 `Phy`, `train_exp23`, or DAVIS evaluator process was found in
the audited process list. This phase must not start any new Exp23 morphology
sweep and must not modify Exp23 checkpoints/evaluations/PRD.

## Existing Source Files

Found:

- `PRD/47_exp25_vor_or_preference_data.md`
- `PRD/45_exp23_two_stage_pool_morphology_sweep.md`
- `PRD/29_videopainter_dpo_adapter_trainer.md`
- `PRD/28_videopainter_adapter_gate2000_result.md`
- `exp14_adapter_videopainter/`
- `training/dpo/`
- `DPO_finetune/`
- `inference/run_OR.py`
- `inference/run_BR.py`
- `inference/metrics.py`

Not present in the Exp25 worktree:

- `PRD/46_exp24_multibackbone_dpo_adapter.md`
- `exp24_multibackbone_dpo_adapter/`

These missing Exp24 artifacts should be looked up from the appropriate Exp24
branch/worktree before updating the compatibility matrix.

## Decision

Proceed with CPU/I/O-safe Exp25 code and manifest work first. Do not launch GPU
OR loser generation or DPO training until PAI is synced, data semantics are
validated, and the corresponding gate scripts pass.
