# Exp39 Status

Current status: `EXP39_H20_MINIMAX_MIRROR_TRANSFER_ENV_SMOKE_PASSED`

## 2026-06-28 H20 Mirror Readback

- Branch: `research/exp39-h20-minimax-mirror-bf16-20260628`.
- Base: `origin/research/exp38-minimax-full-adapter-breakthrough-20260628`.
- Base HEAD: `06b17c0a4be2cb82d1ffbdf7b6c93406f37a3ff8`.
- H20 SSH access was restored with the `codex-h20-2` key.
- H20 GPU audit showed 8 idle NVIDIA H20 GPUs at connection time.
- H20 old repo was dirty/old and was preserved without checkout/reset/clean.
- H20 GitHub object transfer was unreliable; full and broad sparse checkout
  attempts were stopped and removed after they were confirmed to be this
  session's incomplete clones.
- PAI was inspected read-only only. No PAI process was signaled and no PAI GPU
  was used.
- Latest MiniMax state remains plumbing-positive but quality-negative:
  Exp38 is `MINIMAX_SFT_DPO_RESCUE_10STEP_NEGATIVE`.

Next allowed milestone:

```text
PAI read-only MiniMax asset inventory and H20 environment/weight/GPU audit.
```

Still forbidden:

- H20 MiniMax training before mirror/env/bf16 smoke passes.
- PAI training, PAI GPU use, PAI output mutation, or PAI process signaling.
- 30-step MiniMax continuation.
- RC-FPO.
- Universal-adapter or final-SOTA claims.

## 2026-06-28 PAI Asset Inventory

- Status: `PAI_MINIMAX_ASSET_INVENTORY_COMPLETED_READ_ONLY`.
- Unique manifest-referenced PAI paths: `758`.
- Existing referenced paths: `758`.
- Approximate referenced transfer size: `2.781 GiB`.
- PAI model symlink target is about `2.6G`.
- No PAI process was signaled, no PAI GPU was used, and no PAI file was modified.

## 2026-06-28 H20 Env / Weight / GPU Audit

- Status: `H20_ENV_WEIGHT_GPU_AUDIT_COMPLETED_PARTIAL_WORKTREE_BLOCKED`.
- 8 H20 GPUs were idle during audit.
- BF16 support is available in multiple torch environments.
- `/home/nvme01` has about `367G` free but is `90%` used.
- MiniMax `weights/minimax_remover/current` is missing on H20.
- The bad H20 Exp39 partial clone created by this session was removed after
  audit; the H20 Exp39 worktree is currently absent and must be recreated
  cleanly.
- H20 Exp39 worktree remains blocked until a clean mirror is created.

## 2026-06-28 Transfer Block

- Status: `H20_TRANSFER_BLOCKED_DISK_MARGIN_LT_20_PERCENT`.
- Selected manifest-referenced data is about `2.781 GiB`, but H20
  `/home/nvme01` has only about `10.8%` free space.
- The task requires stopping before copy when disk margin is below `20%`.
- No PAI-to-H20 data migration was started.

Report:

- `reports/exp39_h20_transfer_blocked_by_disk_margin.md`

## 2026-06-29 H20 Storage Cleanup

- Status: `H20_STORAGE_CLEANUP_COMPLETED`.
- `/home/nvme01` changed from `90%` used / `367G` free to `58%` used /
  `1.5T` free.
- Deleted stale caches, old archives, old VideoDPO runs, old VC2 logs, upload
  staging assets, and two obsolete generated-loser directories.
- Preserved H20 model weights, experiments, raw external datasets, project
  source, and `data/h20_mirror/minimax`.
- PAI remained untouched.

Report:

- `reports/exp39_h20_storage_cleanup_20260629.md`

## 2026-06-29 H20 Mirror Transfer / Env Repair

- Status: `EXP39_H20_MINIMAX_MIRROR_TRANSFER_ENV_SMOKE_PASSED`.
- H20 source snapshot recreated at
  `/home/nvme01/H20_Video_inpainting_DPO_exp39_minimax_h20`.
- Selected PAI absolute data, MiniMax official repo, and resolved MiniMax
  weights were transferred to H20 mirror storage.
- H20 mirror `pai_abs` now has `9449` files and is about `5.5G`.
- H20 mirror root is about `12G`.
- Data tar SHA256:
  `05e283fd7313d24fe6fac0c97f0fdd0030a0a22cca2899d0dcf3442ed56be786`.
- Weights/repo tar SHA256:
  `dd35570b2bf0f182ccfabe98d974ae4417d8ab99ff121f836193e3c124135782`.
- H20 MiniMax weight symlink now resolves to the mirrored weight target.
- Eleven MiniMax manifests were copied and rewritten for H20 paths.
- Required manifest path audit passed with `0` missing required paths.
- Optional visual-review/evidence assets remain missing: `1256` full-path refs.
- H20 `wan` environment smoke passed for torch/CUDA/BF16, MiniMax imports,
  required packages, required manifests, and MiniMax weights.

PAI protection incident:

- One over-broad read-only PAI `find` command is still in D-state NAS I/O wait
  at last read-only `ps` check. It was not signaled or killed because PAI
  process signaling is forbidden for this track.

Reports:

- `reports/exp39_h20_mirror_transfer_and_env_repair.md`
- `reports/h20_mirror_runtime/exp39_h20_rewritten_manifest_audit.md`
- `reports/h20_mirror_runtime/exp39_h20_required_manifest_path_audit.md`
- `reports/h20_mirror_runtime/exp39_h20_env_smoke_summary.md`
