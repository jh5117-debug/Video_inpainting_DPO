# Exp20 Git Sync Audit

## Result

HAL and PAI are on the required branch and aligned to the same Git HEAD before continuing Exp20 gates.

| Site | Worktree | Branch | HEAD | Status |
| --- | --- | --- | --- | --- |
| HAL | `/home/hj/H20_Video_inpainting_DPO_exp20_autoresearch` | `research/exp20-adaptive-region-autoresearch-20260619` | `0b7891e5d8fa0bf834f2252201f4aeeeb6da364f` | dirty only with current Exp20 fixes before commit |
| PAI | `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp20_autoresearch` | `research/exp20-adaptive-region-autoresearch-20260619` | `0b7891e5d8fa0bf834f2252201f4aeeeb6da364f` | aligned to branch HEAD |

## Sync Strategy

PAI initially had Exp20 files copied by rsync while the Git HEAD remained on the earlier `b020ace` commit. That state was not accepted for training.

The branch was synchronized safely with a Git bundle:

1. HAL created `/tmp/exp20_branch.bundle` from `research/exp20-adaptive-region-autoresearch-20260619`.
2. The bundle was copied to PAI.
3. PAI existing rsync-only dirty files were preserved in a stash named `exp20_pre_bundle_rsync_state_20260619`.
4. PAI fetched the branch from the bundle.
5. PAI merged `FETCH_HEAD` with `git merge --ff-only FETCH_HEAD`.

No `git reset --hard`, `git clean`, force push, or destructive cleanup was used.

## Source Commit

- Required known commit: `0b7891e5d8fa0bf834f2252201f4aeeeb6da364f`
- Current branch tip at audit time: `0b7891e5d8fa0bf834f2252201f4aeeeb6da364f`

## GPU Safety Observation

PAI GPU0-6 are low-memory/idle candidates at audit time. GPU7 has about 58 GB allocated without a listed compute process and is excluded by default.

## 2026-06-20 Continuation Audit

Before continuing the second Exp20 search phase, HAL queried the remote branch:

- remote branch: `origin/research/exp20-adaptive-region-autoresearch-20260619`
- remote HEAD: `d1a79b83f58efdb9a221ae8a823b740d4ddc8c99`

HAL and PAI are both on the required feature branch and already match the
remote HEAD:

| Site | Worktree | Branch | HEAD | Status |
| --- | --- | --- | --- | --- |
| HAL | `/home/hj/H20_Video_inpainting_DPO_exp20_autoresearch` | `research/exp20-adaptive-region-autoresearch-20260619` | `d1a79b83f58efdb9a221ae8a823b740d4ddc8c99` | clean before the new metric-backfill code |
| PAI | `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp20_autoresearch` | `research/exp20-adaptive-region-autoresearch-20260619` | `d1a79b83f58efdb9a221ae8a823b740d4ddc8c99` | only expected runtime untracked files: `exp20_autoresearch_scale_adaptive_region_dpo/first_wave/`, `exp20_autoresearch_scale_adaptive_region_dpo/results.tsv.lock` |

No GitHub fetch fallback or git bundle was needed for this continuation. No
reset, clean, force push, or destructive workspace action was used.
