# Exp41 H20 MiniMax Parallel Readback

Status: `EXP41_H20_MINIMAX_PARALLEL_READBACK_COMPLETED`

Date: 2026-06-29

## Git Readback

- Branch: `research/exp41-h20-minimax-parallel-bf16-20260629`
- Base branch: `origin/research/exp40-minimax-psnr-safe-rescue-20260628`
- Start HEAD: `ecd82ef8bfefd1efba063d2a240631c1b7230b1d`
- H20 worktree:
  `/home/nvme01/H20_Video_inpainting_DPO_exp41_minimax_h20_parallel`
- HAL/local worktree:
  `/home/hj/H20_Video_inpainting_DPO_exp41_minimax_h20_parallel`

Remote MiniMax branch heads read:

| branch | HEAD |
| --- | --- |
| Exp30 | `f69688fe4ff96c4d4f0dcd308eef69822fc1035b` |
| Exp35 | `fb70266d53f5f9abd5e8d09ef9d2de324a10b7d6` |
| Exp36 | `3cd87e4b1a5b30a369ac3604086b7e31a4f45163` |
| Exp37 | `558c2f263469f4ee6ee46e2a1b26a8082515dded` |
| Exp38 | `06b17c0a4be2cb82d1ffbdf7b6c93406f37a3ff8` |
| Exp39 | `fdd83d64883f4cb526f51f2a62d5d93073cf5533` |
| Exp40 | `ecd82ef8bfefd1efba063d2a240631c1b7230b1d` |

## Files Read

PRDs / registry:

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/54_exp38_minimax_full_adapter_breakthrough.md`
- `PRD/55_exp40_minimax_psnr_safe_rescue.md`
- `experiment_registry/exp38_minimax_full_adapter_breakthrough/status.md`
- `experiment_registry/exp40_minimax_psnr_safe_rescue/status.md`
- Exp39 H20 mirror reports from the Exp39 worktree, because Exp41 is based on
  Exp40 and does not include the Exp39 commit.

Reports:

- `reports/exp38_minimax_failure_taxonomy.md`
- `reports/exp38_minimax_train_overfit_diagnosis.md`
- `reports/exp38_minimax_badnoise_v2_diagnostic_scan.md`
- `reports/exp38_minimax_sft_dpo_rescue_10step.md`
- `reports/exp40_minimax_psnr_safe_readback.md`
- `reports/exp40_r1_sample_level_diagnosis.md`
- `reports/exp40_localdpo_v3_pool.md`
- `reports/exp40_minimax_step0_baseline.md`
- Exp39 mirror/env reports:
  `reports/exp39_h20_mirror_transfer_and_env_repair.md`,
  `reports/h20_mirror_runtime/exp39_h20_env_smoke_summary.md`, and
  `reports/h20_mirror_runtime/exp39_h20_required_manifest_path_audit.md`.

## Required Answers

1. Existing PAI MiniMax results:
   - Exp30/35/36/37/38 established plumbing/trainability but no quality-positive
     adapter.
   - Exp38 R1 heldout13 produced full/mask/boundary/outside PSNR deltas
     `+0.102167` / `+0.117230` / `-0.141510` / `-0.037262`, with `0/13`
     clear visual wins.
   - Exp40 built LocalDPO v3 minimum pool and Step0 baseline; no Exp40 training
     result exists yet.
2. H20 MiniMax data:
   - Exp39 mirror contains `9449` files under `pai_abs`, about `5.5G`.
   - Required Exp30/37/38 training/smoke paths have `0` missing refs.
   - Exp40 LocalDPO v3 manifests exist in Git but still need H20 path/material
     validation in the next data audit.
3. H20 manifests currently known valid:
   - Exp30 Gate64, Exp37 LocalDPO/bad-noise, and Exp38 LocalDPO v2 required
     paths passed Exp39 required-path audit.
   - Exp40 LocalDPO v3 pool is valid on PAI/Git by Exp40 reports, but H20
     mirror readiness is pending Exp41 data audit.
4. H20 weights/env:
   - `weights/minimax_remover/current` resolves inside the H20 mirror.
   - H20 `wan` env passed torch/CUDA/BF16 and MiniMax imports in Exp39.
5. PAI outputs missing on H20:
   - Optional Exp39 review-only assets are incomplete (`1256` refs).
   - Exp40 Step0 baseline outputs and LocalDPO v3 materialized root are not yet
     proven mirrored to H20 and must be checked next.
6. Complementary H20 experiments:
   - H20 should not duplicate PAI blindly. It should prioritize data readiness,
     BF16 safety, official protocol audit, PSNR-safe SFT ladder, and only then
     DPO after SFT if gates pass.
7. Frozen code:
   - MiniMax official repo source files, `inference/metrics.py`, shared trainer,
     and Exp1-Exp40 history are frozen.
8. BF16/SIGFPE risks:
   - likely risk zones are attention backend, VAE fp16/bf16 paths, loss
     reduction dtype, checkpoint replay, and timestep edge cases. No source
     patch is allowed in preflight.
9. H20 GPU plan:
   - GPU0: BF16/DDP preflight/controller.
   - GPU1-2: Lane A SFT ladder.
   - GPU3-4: Lane B SFT->DPO only after Lane A passes.
   - GPU5: Lane C data/protocol/inference worker.
   - GPU6: metrics/eval.
   - GPU7: visual pack/reserve.
10. Success gate:
    - Third-backbone evidence requires shadow full PSNR at least `+0.20 dB`
      over Step0, mask `+0.12`, boundary/outside safe, LPIPS/Ewarp safe, and
      visual review confirming real improvement without fogging/over-erasure.

## PAI Protection

PAI was used read-only only. No PAI GPU was used, no PAI process was signaled,
and no PAI file/worktree/output/runtime state was modified.

The over-broad read-only PAI `find` from the Exp39 transfer session was not
present in the latest read-only `ps` check.

## H20 GPU State

GPU4 had one unrelated non-system compute task and was released with TERM to
PGID `3365988`; no KILL was required. Final GPU0-GPU7 state has no compute
apps.

Report:

- `reports/exp41_h20_gpu_release_audit.md`

## Decision

```text
EXP41_H20_MINIMAX_PARALLEL_READBACK_COMPLETED
```

Next milestone is H20 data/weight completeness audit. No MiniMax training has
been launched by Exp41 readback.
