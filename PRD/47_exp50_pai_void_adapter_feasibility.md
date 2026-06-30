# Exp50 PAI VOID Adapter Feasibility

Date: 2026-06-30

Branch: `research/exp50-pai-void-adapter-feasibility-20260630`

Status: `VOID_REPO_READY`

## Objective

Evaluate VOID as a possible third video-inpainting adapter candidate after ROSE was paused. VOID is prioritized because public sources expose code, model checkpoints, data-generation code, and explicit training scripts.

## Safety Boundaries

- Do not continue ROSE downloads or training.
- Do not modify `inference/metrics.py`.
- Do not modify shared trainer code.
- Do not modify official VOID source.
- Do not use VOR-Eval for training, filtering, or threshold design.
- Do not use hard comp for promotion.
- Do not run 50/100/300/500/1000/2000-step training.
- Do not claim VOID third-backbone positive unless one-step and heldout 10-step micro gates truly pass.

## Paths

- HAL worktree target: `/home/hj/H20_Video_inpainting_DPO_exp50_void_adapter`
- PAI worktree fallback in use: `/home/hj/H20_Video_inpainting_DPO_exp50_void_adapter`
- Requested primary PAI worktree: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp50_void_adapter` (not writable by `hj` at readback)
- Output root: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp50_pai_void_adapter_feasibility`
- Log root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp50_pai_void_adapter_feasibility`
- Runtime root: `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp50_pai_void_adapter_feasibility`
- Third-party root: `/mnt/nas/hj/H20_Video_inpainting_DPO/third_party/VOID`
- Weights root: `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/void`
- Data root: `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/void`

## Milestone A Readback

Generated:

- `reports/exp50_void_readback.md`
- `reports/exp50_void_public_source_audit.md`
- `reports/exp50_void_public_source_audit.csv`
- `reports/exp50_void_readback_summary.json`

Key result: VOID has stronger adapter feasibility than ROSE because its public repo documents `data_generation/`, `scripts/cogvideox_fun/train_void.sh`, `scripts/cogvideox_fun/train_void_warped_noise.sh`, quadmask conditioning, and Pass1/Pass2 checkpoints. No download was performed before this readback commit.

## Permission Caveat

At readback, `hj` can write logs/runtime but cannot create asset/output directories under several requested NAS roots. Milestone B must first resolve directory ownership or use an explicitly approved fallback.


## Milestone B Update - 2026-06-30

Status: `VOID_ASSETS_BLOCKED`.

Pre-download disk and permission checks were run. No VOID official repo, Pass1/Pass2 weights, CogVideoX-Fun base model, or sample data was downloaded because the required NAS target roots are not writable by `hj`, and neither passwordless sudo nor root SSH is available from this session. The blocked target roots are `/third_party/VOID`, `/weights/void`, `/data/external/void`, and `/experiments/dpo/exp50_pai_void_adapter_feasibility` under `/mnt/nas/hj/H20_Video_inpainting_DPO`.

The task is blocked until those directories are created/chowned for `hj`, or an explicit alternate asset/output root is approved.


## Milestone B0 Update - 2026-06-30

Status: `VOID_ASSET_PERMISSION_RECOVERED`.

Using `hj` identity, Codex verified read/write/execute access and write-probe behavior for all six EXP50 directories after the user-provided root-side minimal permission fix. Codex did not run chmod/chown or any root permission script.

Reports:

- `reports/exp50_void_permission_recovery.md`
- `reports/exp50_void_permission_recovery.csv`


## Milestone B1 Update - 2026-06-30

Status: `VOID_REPO_READY`.

The official VOID repository was cloned/audited at `/mnt/nas/hj/H20_Video_inpainting_DPO/third_party/VOID/Netflix_void-model`. Reports:

- `reports/exp50_void_repo_audit.md`
- `reports/exp50_void_repo_audit.csv`
- `reports/exp50_void_repo_audit_summary.json`
- `reports/exp50_void_repo_inventory.txt`

No official VOID source was modified.
