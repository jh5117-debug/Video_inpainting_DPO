# Exp39 H20 Storage Cleanup

Date: 2026-06-29 H20 local time

Status: `H20_STORAGE_CLEANUP_COMPLETED`

## Scope

Cleanup was performed only on H20. PAI was not touched.

No H20 training was launched, no GPU process was modified, and no PAI process
was signaled.

## Disk Result

Before cleanup:

```text
/home/nvme01: 3.4T total, 3.1T used, 367G free, 90% used
```

After cleanup:

```text
/home/nvme01: 3.4T total, 2.0T used, 1.5T free, 58% used
```

The H20 disk margin is now above the 20% transfer threshold.

## Deleted Targets

First wave:

| path | size before |
| --- | ---: |
| `/home/nvme01/.cache/huggingface/hub` | 62G |
| `/home/nvme01/H20_Video_inpainting_DPO/.wandb_cache` | 6.6G |
| `/home/nvme01/H20_Video_inpainting_DPO/archives` | 241G |
| `/home/nvme01/VideoDPO_runs` | 163G |
| `/home/nvme01/H20_Video_inpainting_DPO/logs/videodpo_vc2_dpo` | 167G |
| `/home/nvme01/H20_Video_inpainting_DPO/logs/vc2_dpo_videoinpainting_h20_gpu2-7_20260507_103803` | 39G |
| `/home/nvme01/H20_Video_inpainting_DPO/logs/vc2_dpo_videoinpainting_h20_gpu2-7_20260507_211147` | 14G |

Second wave:

| path | size before |
| --- | ---: |
| `/home/nvme01/H20_Video_inpainting_DPO/data/hf_upload` | 111G |
| `/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4` | 249G |
| `/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/exp07_fix_videodpo_smallmask15_20_prior_k4` | 74G |

## Preserved

- `/home/nvme01/H20_Video_inpainting_DPO/weights` remains present, about 93G.
- `/home/nvme01/H20_Video_inpainting_DPO/experiments` remains present, about 363G.
- `/home/nvme01/H20_Video_inpainting_DPO/data/h20_mirror/minimax` remains present.
- Raw external datasets and current project source were not deleted.
- Exp39 H20 mirror remains pending because the clean worktree still needs to be
  recreated.

## H20 Cleanup Logs

```text
/home/nvme01/cleanup_exp39_h20_20260629_023536.log
/home/nvme01/cleanup_exp39_h20_data_20260629_023617.log
```

## Decision

```text
H20_STORAGE_CLEANUP_COMPLETED
```

The previous transfer block caused by disk margin is cleared from a capacity
standpoint. Code mirror and MiniMax weight-symlink readiness still need to be
fixed before H20 MiniMax smoke or training.
