# Exp39 H20 MiniMax Mirror Transfer And Env Repair

Date: 2026-06-28/2026-06-29

Status: `EXP39_H20_MINIMAX_MIRROR_TRANSFER_ENV_SMOKE_PASSED`

## Protection

- PAI was used only for read-only SSH/rsync-from/ps inspection.
- No PAI GPU was used.
- No PAI files, worktrees, checkpoints, locks, outputs, or heartbeats were
  modified.
- No MiniMax training, inference run, optimizer step, or adapter continuation
  was launched.
- `inference/metrics.py` and shared trainers were not modified.

Incident note:

- One over-broad read-only PAI `find` command remains in D-state NAS I/O wait
  as of the last read-only `ps` check: PID `2044802`, PGID `2044801`, elapsed
  about `30:37`.
- It was not signaled or killed because this track's PAI protection rule
  forbids sending signals to PAI processes. No further broad PAI filesystem
  scans were launched after this was identified.

## H20 Source Mirror

- Source snapshot was extracted at:
  `/home/nvme01/H20_Video_inpainting_DPO_exp39_minimax_h20`
- Source archive SHA256:
  `63b993473027c18128b4f18ba1a1842f0270be377a2a4372bed71f6183fa38ed`
- Local Exp39 branch remains the commit source:
  `research/exp39-h20-minimax-mirror-bf16-20260628`

## Data / Weight Transfer

- H20 mirror root:
  `/home/nvme01/H20_Video_inpainting_DPO/data/h20_mirror/minimax`
- H20 `pai_abs` file count: `9449`
- H20 `pai_abs` size: `5.5G`
- H20 mirror root size: `12G`
- H20 `/home/nvme01` remained at about `1.5T` free / `58%` used after transfer.

Transferred archives:

| archive | SHA256 | result |
| --- | --- | --- |
| `exp39_minimax_pai_abs_20260628.tar` | `05e283fd7313d24fe6fac0c97f0fdd0030a0a22cca2899d0dcf3442ed56be786` | verified/extracted |
| `exp39_minimax_weights_repo_pai_abs_20260628.tar` | `dd35570b2bf0f182ccfabe98d974ae4417d8ab99ff121f836193e3c124135782` | verified/extracted |

MiniMax weight symlink repaired:

```text
/home/nvme01/H20_Video_inpainting_DPO/weights/minimax_remover/current
-> /home/nvme01/H20_Video_inpainting_DPO/data/h20_mirror/minimax/pai_abs/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/minimax
```

## Manifest Rewrite

- Original manifests:
  `/home/nvme01/H20_Video_inpainting_DPO/data/h20_mirror/minimax/manifests/pai_original`
- H20 rewritten manifests:
  `/home/nvme01/H20_Video_inpainting_DPO/data/h20_mirror/minimax/manifests/h20_rewritten`
- Manifest count: `11`
- Total rows: `713`
- Total absolute refs before rewrite: `4496`
- Full-path audit status: `H20_MANIFEST_REWRITE_HAS_MISSING_PATHS`
- Missing full-path refs: `1256`
- Required-path audit status: `H20_REQUIRED_MANIFEST_PATHS_COMPLETE`
- Required missing paths: `0`

The missing full-path refs are optional review/evidence assets such as review
sheets, temporal strips, side-by-side videos, and diagnostic comps. Required
training/smoke paths such as condition, winner, mask, loser/raw frames, raw mp4,
and bad-noise/state paths are complete.

Synced H20 runtime reports:

- `reports/h20_mirror_runtime/exp39_h20_rewritten_manifest_audit.md`
- `reports/h20_mirror_runtime/exp39_h20_rewritten_manifest_audit.csv`
- `reports/h20_mirror_runtime/exp39_h20_rewritten_manifest_summary.json`
- `reports/h20_mirror_runtime/exp39_h20_required_manifest_path_audit.md`
- `reports/h20_mirror_runtime/exp39_h20_required_manifest_path_audit.csv`

## Environment Smoke

- Python: `/home/nvme01/miniconda3/envs/wan/bin/python`
- Torch: `2.5.1+cu124`
- CUDA runtime: `12.4`
- H20 BF16 supported: `true`
- MiniMax imports passed:
  - `pipeline_minimax_remover`
  - `transformer_minimax_remover`
- Required packages passed:
  `diffusers`, `transformers`, `safetensors`, `accelerate`, `cv2`, `decord`
- Required weight files passed:
  - transformer safetensors: `2254157576` bytes
  - VAE safetensors: `507591892` bytes
  - transformer/vae/scheduler config JSON files

Smoke status:

```text
H20_MINIMAX_ENV_SMOKE_PASSED
```

Synced H20 smoke reports:

- `reports/h20_mirror_runtime/exp39_h20_env_smoke_summary.md`
- `reports/h20_mirror_runtime/exp39_h20_env_smoke_summary.json`
- `reports/h20_mirror_runtime/exp39_h20_env_smoke_checks.csv`

## Decision

The H20 MiniMax mirror is now usable for later explicitly authorized debugging
or smoke work. This milestone does not authorize MiniMax training, 30-step
continuation, RC-FPO, universal-adapter claims, or final-SOTA claims.

```text
EXP39_H20_MINIMAX_MIRROR_TRANSFER_ENV_SMOKE_PASSED
```
