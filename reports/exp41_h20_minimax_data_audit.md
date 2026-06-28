# Exp41 H20 MiniMax Data / Weight Audit

Status: `H20_MINIMAX_DATA_READY`

Date: 2026-06-29 H20 / 2026-06-28 UTC.

This audit validates the H20 MiniMax mirror for Exp41 without launching training.
PAI was used only as a read-only rsync source. No PAI GPU, process signal,
file mutation, output mutation, or runtime mutation was performed.

## Decision

`H20_MINIMAX_DATA_READY` is passed for the MiniMax data/weight mirror needed by
Exp41 preflight. This is a data readiness gate only, not a MiniMax adapter
quality-positive claim.

Training remains locked behind BF16/SIGFPE runtime preflight and official
MiniMax protocol audit.

## H20 Mirror

- Mirror root: `/home/nvme01/H20_Video_inpainting_DPO/data/h20_mirror/minimax/pai_abs`.
- Post-transfer file count: `14899`.
- Post-transfer apparent size: `7.8G`.
- Latest H20 GPU query: GPU0 `28 MiB`, GPU1-GPU7 `1 MiB`, utilization `0%`,
  compute apps `0`.

## Transfers Completed

| transfer | files | bytes | source | destination | status |
| --- | ---: | ---: | --- | --- | --- |
| Exp40 LocalDPO v3 evidence | 224 | 333063059 | PAI `/mnt/nas/hj/...` read-only | H20 mirror `pai_abs/...` | PASS |
| Legacy Exp30/37/38 evidence | 232 | 237644031 | PAI `/mnt/nas/hj/...` read-only | H20 mirror `pai_abs/...` | PASS |

The earlier mirror already contained the core VOR source data, materialized
frame directories, LocalDPO loser frame directories, and MiniMax weights. The
new transfers filled missing raw/review/side-by-side/temporal evidence assets.

## Manifests Validated

| manifest | active refs checked | missing | status |
| --- | ---: | ---: | --- |
| exp30_heldout16_v3 | 134 | 0 | PASS |
| exp30_train32_v3 | 240 | 0 | PASS |
| exp37_badnoise_states | 128 | 0 | PASS |
| exp37_heldout16 | 96 | 0 | PASS |
| exp37_train32 | 192 | 0 | PASS |
| exp38_badnoise_v2_train30 | 120 | 0 | PASS |
| exp38_heldout13_filtered | 78 | 0 | PASS |
| exp38_train30_filtered | 180 | 0 | PASS |
| exp41_exp40_search_h20 | 225 | 0 | PASS |
| exp41_exp40_shadow_h20 | 231 | 0 | PASS |
| exp41_exp40_train_h20 | 618 | 0 | PASS |

Total active refs checked: `2242`. Final missing refs: `0`.

## Key Coverage

| key | refs checked |
| --- | ---: |
| `condition_frame_dir` | 112 |
| `condition_mp4_direct` | 22 |
| `condition_path` | 313 |
| `diagnostic_comp_mp4` | 26 |
| `loser_path` | 313 |
| `mask_frame_dir` | 112 |
| `mask_mp4_direct` | 22 |
| `mask_path` | 313 |
| `raw_output_mp4` | 244 |
| `review_sheet` | 236 |
| `side_by_side_mp4` | 41 |
| `temporal_strip_16` | 41 |
| `winner_frame_dir` | 112 |
| `winner_mp4_direct` | 22 |
| `winner_path` | 313 |

## Decode Audit

`imageio.v3` was available on H20; system `ffprobe`/`ffmpeg` and `cv2` were not
available, so decode validation used first-frame `imageio.v3.imread`.

| item | count | failures | notes |
| --- | ---: | ---: | --- |
| Exp40 raw_output_mp4 | 112 | 0 | all train/search/shadow raw outputs readable |
| Exp40 condition_mp4_direct | 22 | 0 | de-duplicated VOR source mp4s readable |
| Exp40 winner_mp4_direct | 22 | 0 | de-duplicated VOR winner mp4s readable |
| Exp40 mask_mp4_direct | 22 | 0 | first frame non-empty for every checked mask |

## Weight Audit

H20 symlink:

```text
/home/nvme01/H20_Video_inpainting_DPO/weights/minimax_remover/current
-> /home/nvme01/H20_Video_inpainting_DPO/data/h20_mirror/minimax/pai_abs/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/minimax
```

Files resolved on H20:

| file | bytes |
| --- | ---: |
| scheduler/scheduler_config.json | 751 |
| transformer/config.json | 422 |
| transformer/diffusion_pytorch_model.safetensors | 2254157576 |
| vae/config.json | 724 |
| vae/diffusion_pytorch_model.safetensors | 507591892 |

## Generated Evidence

- `reports/exp41_h20_minimax_data_audit.csv`
- `reports/exp41_h20_minimax_manifest_validation.csv`
- `reports/exp41_h20_minimax_missing_assets.csv`
- `reports/exp41_h20_minimax_decode_audit.csv`

## Remaining Gates

- `H20_MINIMAX_BF16_SAFE_READY` or a precise blocked/fp32-only state.
- `H20_MINIMAX_PROTOCOL_MATCHES_OFFICIAL`.
- No SFT, DPO, or 500-step confirmation is authorized by this data gate alone.
