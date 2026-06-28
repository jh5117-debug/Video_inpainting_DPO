# Exp39 H20 MiniMax Mirror + BF16/SIGFPE Debug

Status: `EXP39_H20_MINIMAX_MIRROR_TRANSFER_ENV_SMOKE_PASSED`

Exp39 is an isolated H20 mirror and runtime-debug track for the MiniMax
experiments from Exp30/35/36/37/38. It does not take over PAI training and does
not modify PAI worktrees, outputs, locks, checkpoints, or running processes.

## Ground Rules

- PAI is read-only for this track.
- Do not send signals to PAI processes.
- Do not use PAI GPUs.
- Do not launch new MiniMax training on PAI.
- Do not modify PAI Exp30/35/36/37/38/31/33 worktrees or outputs.
- Do not modify `inference/metrics.py` or shared trainers.
- Do not write universal-adapter, final-SOTA, all-models-supported, or
  top-conference-novelty-confirmed claims.
- H20 writes must stay under the Exp39 branch, H20 mirror root, or Exp39
  output/log/runtime roots.

## 2026-06-28 Readback

- Branch: `research/exp39-h20-minimax-mirror-bf16-20260628`.
- Local readback worktree:
  `/home/hj/H20_Video_inpainting_DPO_exp39_minimax_h20_local`.
- Intended H20 worktree:
  `/home/nvme01/H20_Video_inpainting_DPO_exp39_minimax_h20`.
- Base branch:
  `origin/research/exp38-minimax-full-adapter-breakthrough-20260628`.
- Base HEAD read back: `06b17c0a4be2cb82d1ffbdf7b6c93406f37a3ff8`.
- H20 old repo was dirty and old, so it was not checked out or reset.
- H20 GitHub object transfer was too slow/unreliable for a full or broad sparse
  checkout in this milestone. Broken partial clones were stopped and removed
  when they were confirmed to be created by this Exp39 session.
- PAI MiniMax processes were inspected read-only only; no PAI signals were sent.

Latest MiniMax scientific state:

- Exp30: `MINIMAX_ADAPTER_RECIPE_NOT_READY`.
- Exp35: `MINIMAX_RESCUE_RECIPE_NOT_READY`.
- Exp36: `MINIMAX_PLUMBING_POSITIVE_RECIPE_NOT_READY`.
- Exp37: `MINIMAX_LOCALDPO_BADNOISE_PARETO_MIXED`, but paper role remains
  plumbing-only.
- Exp38: `MINIMAX_SFT_DPO_RESCUE_10STEP_NEGATIVE`.

Decision:

```text
H20 mirror setup may continue with asset inventory and small-file transfer.
No H20 MiniMax training is authorized until the code mirror, manifests, weights,
environment audit, and bf16/SIGFPE smoke all pass.
```

Report:

- `reports/exp39_h20_minimax_mirror_readback.md`

## 2026-06-29 H20 Mirror Transfer / Env Repair

- Status: `EXP39_H20_MINIMAX_MIRROR_TRANSFER_ENV_SMOKE_PASSED`.
- H20 source snapshot exists at
  `/home/nvme01/H20_Video_inpainting_DPO_exp39_minimax_h20`.
- H20 mirror root is
  `/home/nvme01/H20_Video_inpainting_DPO/data/h20_mirror/minimax`.
- `pai_abs` contains `9449` files and is about `5.5G`.
- Data archive SHA256:
  `05e283fd7313d24fe6fac0c97f0fdd0030a0a22cca2899d0dcf3442ed56be786`.
- Weight/repo archive SHA256:
  `dd35570b2bf0f182ccfabe98d974ae4417d8ab99ff121f836193e3c124135782`.
- MiniMax `weights/minimax_remover/current` was repaired to point inside the
  H20 mirror.
- Eleven MiniMax manifests were copied and rewritten for H20 paths.
- Full manifest path audit has `1256` missing optional review/evidence assets,
  but the required-path audit has `0` missing training/smoke paths.
- H20 `wan` environment smoke passed for torch/CUDA/BF16, MiniMax module
  imports, required packages, required manifests, and MiniMax weight files.

PAI protection note:

- PAI remained read-only.
- One over-broad read-only PAI `find` command from this session remained in
  D-state NAS I/O wait at last `ps` readback. It was not signaled or killed to
  preserve the explicit PAI no-signal rule.

Reports:

- `reports/exp39_h20_mirror_transfer_and_env_repair.md`
- `reports/h20_mirror_runtime/exp39_h20_env_smoke_summary.md`
- `reports/h20_mirror_runtime/exp39_h20_required_manifest_path_audit.md`
