# Exp27 Official Paper-Code Cache Sync

Date: 2026-06-23

## Scope

Synced pinned official paper-code repositories from HAL to PAI read-only cache:

`/mnt/nas/hj/video_dpo_paper_code_cache/`

The official cache is not committed to Git. Exp27 adaptation code remains under
`exp27_paper_grounded_preference_study/`.

## Synced Repositories

| repo | commit | PAI path |
| --- | --- | --- |
| Local-DPO | `7528e966b17283cfa638577827e456737335f030` | `/mnt/nas/hj/video_dpo_paper_code_cache/Local-DPO_7528e966b17283cfa638577827e456737335f030` |
| Diffusion-SDPO | `84fb241c1b89705a247da8b0d6047798ca49830d` | `/mnt/nas/hj/video_dpo_paper_code_cache/Diffusion-SDPO_84fb241c1b89705a247da8b0d6047798ca49830d` |
| Linear-DPO | `663179c7adbbbd2d77b97b5841534447eb291ebd` | `/mnt/nas/hj/video_dpo_paper_code_cache/Linear-DPO_663179c7adbbbd2d77b97b5841534447eb291ebd` |

Compatibility symlinks were created under:

`/mnt/nas/hj/video_dpo_paper_code_cache/repos/`

so Exp27 can use:

`EXP27_PAPER_CODE_ROOT=/mnt/nas/hj/video_dpo_paper_code_cache/repos`

## PAI CPU Parity Result

Runtime snapshot:

`/mnt/workspace/hj/nas_hj/runtime_code_snapshots/exp27_worktree_cache_sync_latest`

Output:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp27_paper_grounded_preference_study/pai_cpu_parity_cache_sync_latest`

Result:

`EXP27_CPU_PRIMITIVE_PARITY_PASSED`

- LocalDPO official mask generation: passed after a narrow runtime compatibility shim for modern Matplotlib RGB bytes and missing `cv2` module binding.
- LocalDPO latent fusion / outside reinjection primitive: passed.
- Diffusion-SDPO official lambda extraction: passed, `max_abs_diff=0.0`.
- Linear-DPO primitive and EMA update: passed, `ema_max_abs_diff=0.0`.

The compatibility shim is local to Exp27. It does not modify the official cache.

## Still Pending

- Real DiffuEraser batch parity for SDPO.
- Real DiffuEraser batch parity for Linear-DPO Frozen and EMA.
- LocalDPO complete data + original loss baseline.
- Any 1/10/50 study.
- RC-FPO gate.
