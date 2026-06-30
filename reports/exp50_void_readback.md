# Exp50 VOID Readback

## Environment

- Host: PAI `dsw-753014-85f54df947-bkp7h` verified.
- Branch: `research/exp50-pai-void-adapter-feasibility-20260630`
- Start HEAD: `34844d75aba585542b311098417f67c7274f6434` from `origin/main`.
- Primary PAI worktree candidate `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp50_void_adapter` is not writable for user `hj`; fallback worktree is `/home/hj/H20_Video_inpainting_DPO_exp50_void_adapter`.
- NAS permission caveat: `hj` can write logs/runtime but cannot currently create under `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo`, `third_party`, `weights`, or `data/external`. Asset/download milestones will require those target directories to be created/chowned or an approved fallback.

## Current Project Readback

- DiffuEraser and VideoPainter remain the main positive adapter evidence.
- MiniMax remains protocol/plumbing-positive but adaptation recipe negative; it is not third-backbone evidence.
- ROSE is paused: Exp49 showed official inference baseline/loser-generator signal, but training-forward is blocked because public training runner/loss/target plumbing is unclear.
- VOID is prioritized because public README/model cards explicitly expose inference, quadmask input format, data-generation code, Pass1/Pass2 checkpoints, and training scripts.

## Answers Required By Milestone A

1. Why pause ROSE? Because Exp49 found ROSE useful as baseline/loser-generator but `ROSE_TRAINING_FORWARD_BLOCKED`; public trainable objective/runner is unclear.
2. Why VOID is a stronger adapter candidate? VOID public sources expose `data_generation/`, `datasets/void_train_data.json`, `scripts/cogvideox_fun/train_void.sh`, `scripts/cogvideox_fun/train_void_warped_noise.sh`, and model cards documenting Pass1/Pass2 and quadmask conditioning.
3. Which training scripts exist? `scripts/cogvideox_fun/train_void.sh` for Pass1 and `scripts/cogvideox_fun/train_void_warped_noise.sh` for Pass2 warped-noise refinement, per official README.
4. What assets are needed? Netflix VOID official repo, `void_pass1.safetensors`, optional `void_pass2.safetensors`, `alibaba-pai/CogVideoX-Fun-V1.5-5b-InP`, included sample data, and later VOR-Train converted quadmask data.
5. What hardware is expected? HF/README note 40GB+ VRAM for notebook/inference and training used 8x A100 80GB with DeepSpeed ZeRO stage 2. PAI H20/L20X feasibility needs smoke verification.
6. How does VOID differ from VOR-OR? VOR-OR removes an object against a clean background target; VOID explicitly models interaction deletion and affected regions using quadmask values for primary, overlap, affected, and keep regions.
7. What is the VOR-to-VOID adapter plan? Convert `V_obj` to `input_video.mp4`/`rgb_full.mp4`, `V_bg` to `rgb_removed.mp4`, object mask to quadmask primary value 0, `abs(V_obj - V_bg)` affected region to value 127, overlap to 63, background to 255, plus `prompt.json` with a clean-background description.
8. What is the promotion gate? Assets/env -> code trainable-forward audit -> VOR quadmask Gate8 -> official inference Gate8/Gate16 -> zero-gap/one-step only if trainable-forward feasible -> 10-step LoVI-DPO only if one-step passes.
9. What remains forbidden? ROSE continuation, VOR-Eval training/filtering, hard comp, long training, modifying shared trainer/metrics/official VOID source, and third-backbone/universal/SOTA claims without true heldout micro evidence.

## Milestone A Status

`EXP50_VOID_READBACK_COMPLETED_WITH_NAS_PERMISSION_CAVEAT`

No VOID code/model/base download was performed before this readback commit.
