# Exp43 H20 Stage2 SFT Runner Readback

Status: `H20_EXP43_STAGE2_SFT_RUNNER_READBACK_COMPLETED`

Date: 2026-06-29.

Branch: `research/exp43-h20-minimax-stage2-sft-runner-20260629`

Start HEAD: `03ce2eb5fdc476789280eaea97f2145a0aa369b5`

Base branch: `origin/research/exp41-h20-minimax-parallel-bf16-20260629`

## Git Readback

Readback commands completed before this milestone:

- `git fetch --all --prune`
- `git branch --show-current`
- `git rev-parse HEAD`
- `git status --short`
- `git log -12 --oneline`
- `git diff --stat`
- `git diff --check`

The local Exp43 worktree was clean at the start of readback. H20 worktree
`/home/nvme01/H20_Video_inpainting_DPO_exp43_minimax_stage2_sft` was also on
`03ce2eb5fdc476789280eaea97f2145a0aa369b5` and clean.

## Files Read

Exp43 readback read the current PRD and registry files:

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/55_exp40_minimax_psnr_safe_rescue.md`
- `PRD/56_exp41_h20_minimax_parallel_bf16.md`
- `PRD/57_exp43_h20_minimax_stage2_sft_runner.md`
- `experiment_registry/exp40_minimax_psnr_safe_rescue/status.md`
- `experiment_registry/exp41_h20_minimax_parallel_bf16/status.md`
- `experiment_registry/exp43_h20_minimax_stage2_sft_runner/status.md`

Reports read:

- `reports/exp41_h20_minimax_data_audit.md`
- `reports/exp41_h20_bf16_preflight.md`
- `reports/exp41_h20_bf16_preflight_summary.json`
- `reports/exp41_h20_minimax_official_protocol_audit.md`
- `reports/exp41_h20_official_vs_current_visual_review.csv`
- `reports/exp41_h20_sft_badnoise_ladder.md`
- `reports/exp41_h20_sft_ladder_patch_proposal.md`
- `reports/exp43_h20_gpu_release_audit.md`

Exp39 mirror reports requested by the prompt are not present in this Exp41-based
branch:

- `reports/exp39_h20_mirror_transfer_and_env_repair.md`
- `reports/h20_mirror_runtime/exp39_h20_env_smoke_summary.md`
- `reports/h20_mirror_runtime/exp39_h20_required_manifest_path_audit.md`

This is a report-history gap, not a current data gap. Exp41 independently
validated the H20 MiniMax mirror, weights, decode path, official protocol, and
BF16 runtime before Exp43 was created.

Exp42 was read locally and read-only from
`/home/hj/H20_Video_inpainting_DPO_exp42_pai_minimax_data`. Latest local Exp42
HEAD was `b8c90a4fa81ea636d123614bd3238e0be1a433ca`, status
`EXP42_PAI_MINIMAX_DATA_READBACK_COMPLETED`. Exp42 has not yet provided a
successful-removal pseudo-success pool to H20; it owns PAI-side mining and stays
separate from Exp43.

## 1. What Blocked Exp41

Exp41 reached `H20_MINIMAX_SFT_BLOCKED`, not because H20 lacked data, weights,
BF16 support, or official protocol identity. The blocker was runner scope:

- Exp35 winner-SFT runner caps at 10 steps.
- Exp36 winner-SFT S0/S1 runner caps at 10 steps.
- Exp35 rescue DPO runner caps at 10 steps and is not SFT-only.
- Exp37 LocalDPO bad-noise runner caps DPO at 10 steps and SFT warmup at
  5 steps.

Lane A required a real 30/100/300-step SFT-only ladder. Under the previous
no-source-change rule, Exp41 could only write a patch proposal and stop.

## 2. Authorized Exp43 Runner

This prompt explicitly authorizes a new isolated runner under:

```text
exp43_h20_minimax_stage2_sft_runner/
```

The runner may add Exp43-only code, configs, launchers, tests, reports, and
manifests. It must not modify shared trainers, `inference/metrics.py`, MiniMax
official source, or Exp1-Exp42 historical results.

The intended runner is a Stage2-style SFT ladder that can run true 30/100/300
optimizer steps, and conditional DPO/500-step gates only if prior gates pass.

## 3. Data Present on H20

Exp41 data audit passed as `H20_MINIMAX_DATA_READY`:

- H20 mirror root:
  `/home/nvme01/H20_Video_inpainting_DPO/data/h20_mirror/minimax`.
- Active refs checked: `2242`.
- Missing refs: `0`.
- Exp40 H20-safe LocalDPO v3 manifests:
  - train: `64` rows.
  - search: `24` rows.
  - shadow: `24` rows.
- Decode audit passed for `112` Exp40 raw outputs and de-duplicated source,
  winner, and mask mp4s.
- VOR-Eval is not included and remains excluded.

Exp43 will build from H20 mirrored data first. If Exp42 later produces
successful-removal or bad-noise data, it can be pulled read-only only after
checksums, path rewrite, and decode validation.

## 4. Weights and Environment on H20

H20 runtime paths verified:

- MiniMax mirror size: about `14G`.
- `wan` environment size: about `12G`.
- Exp43 H20 worktree size: about `263M`.
- MiniMax weight symlink:
  `/home/nvme01/H20_Video_inpainting_DPO/weights/minimax_remover/current`.
- Torch/CUDA from H20 smoke: `2.5.1+cu124` / `12.4`.
- CUDA device count: `8`.
- BF16 supported: true.

Resolved MiniMax weight files from Exp41:

- `scheduler/scheduler_config.json`
- `transformer/config.json`
- `transformer/diffusion_pytorch_model.safetensors`
- `vae/config.json`
- `vae/diffusion_pytorch_model.safetensors`

## 5. BF16 Risks

Historical H20 failures included SIGFPE risk under unsafe mixed precision. Exp41
preflight reduced the immediate risk but does not remove the need for an Exp43
runner-local policy.

Exp41 P0-P7 passed, including DDP8 one-batch MiniMax training, with:

- no SIGFPE;
- no OOM;
- no CUDA error;
- no NaN/Inf;
- no Xid;
- finite losses and gradients;
- rank0 checkpoint save/reload for train cases.

Exp43 still must record its own resolved precision config and use the safe
policy: VAE fp32, DiT bf16 autocast, fp32 loss/reductions, safe attention
backend flags, timestep clamp away from exact 0/1, fp32 gradient norm, and no
silent fallback.

## 6. 8-GPU Plan

H20 GPU0-GPU7 were released for Exp43 after audit. Initial Exp43 GPU audit found
no compute apps; no PIDs/PGIDs were killed.

Planned use:

1. Implement Exp43 isolated runner and single-GPU preflight.
2. Run Exp43 P0-P7 BF16-safe preflight, including DDP2 and DDP8.
3. Use DDP8 only after P7 passes for Exp43.
4. Run 30-step SFT for all legal recipes/LRs.
5. Promote only passing recipes to 100-step, and only passing 100-step recipes
   to 300-step.
6. Run DPO after SFT and 500-step confirmation only if their gates unlock.

## 7. Step Gates

SFT 30-to-100 gate:

- search full PSNR >= +0.08;
- mask PSNR >= +0.05;
- boundary PSNR >= -0.02;
- outside PSNR >= -0.02;
- LPIPS not worse by more than 0.001;
- Ewarp not worse by more than 0.05;
- visual worse <= 25%;
- no fogging or over-erasure.

SFT 100-to-300 gate:

- shadow full PSNR >= +0.15;
- mask PSNR >= +0.10;
- boundary PSNR >= 0;
- outside PSNR >= 0;
- LPIPS and Ewarp safe;
- visual better >= 30%;
- worse <= 20%.

DPO and 500-step gates remain locked until SFT passes. No 1000/2000-step run is
authorized by this prompt.

## 8. What Must Not Be Modified

Exp43 must not modify:

- PAI worktrees, outputs, GPUs, or processes;
- MiniMax official source;
- shared trainers;
- `inference/metrics.py`;
- Exp1-Exp42 historical results;
- VOR-Eval data for training, selection, or tuning.

Exp43 also must not write universal-adapter, final-SOTA, or top-conference
novelty claims.

## 9. Difference From PAI Exp42

Exp42 is PAI-side data mining:

- official MiniMax successful-removal mining;
- bad-noise state construction;
- Stage2-style data gates;
- short gated training only after PAI data gates.

Exp43 is H20-side runner/system work:

- isolated SFT ladder runner;
- BF16-safe DDP8 training path;
- true 30/100/300-step H20 ladder;
- conditional DPO and 500-step confirmation if gates pass.

Exp43 may consume Exp42 outputs only read-only after they exist and pass
checksum/decode/path validation. As of this readback, Exp42 is readback-only and
does not yet unlock pseudo-success distillation on H20.

## Decision

`H20_EXP43_STAGE2_SFT_RUNNER_READBACK_COMPLETED`.

Next milestone: implement Exp43-only BF16-safe runner preflight files under
`exp43_h20_minimax_stage2_sft_runner/`, then run Exp43 P0-P7 before any ladder
training.
