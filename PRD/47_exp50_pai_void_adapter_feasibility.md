# Exp50 PAI VOID Adapter Feasibility

Date: 2026-06-30

Branch: `research/exp50-pai-void-adapter-feasibility-20260630`

Status: `EXP50_VOID_READBACK_COMPLETED_WITH_NAS_PERMISSION_CAVEAT`

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
