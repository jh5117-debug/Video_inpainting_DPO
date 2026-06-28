# Exp39 PAI MiniMax Asset Inventory

Date: 2026-06-28

Status: `PAI_MINIMAX_ASSET_INVENTORY_COMPLETED_READ_ONLY`

PAI was accessed read-only. No PAI process was signaled, no PAI GPU was used,
and no PAI file was modified.

## Manifest Inventory

| manifest | status | rows | absolute path refs |
| --- | --- | ---: | ---: |
| `exp30_vor_or_multimodel_minimax/manifests/vor_or_gate64_train32_v3.jsonl` | OK | 32 | 153 |
| `exp30_vor_or_multimodel_minimax/manifests/vor_or_gate64_heldout16_v3.jsonl` | OK | 16 | 80 |
| `exp30_vor_or_multimodel_minimax/manifests/vor_or_gate64_selected_primary_v3.jsonl` | OK | 50 | 243 |
| `exp30_vor_or_multimodel_minimax/manifests/vor_or_gate64_candidates_all_v3.jsonl` | OK | 256 | 1152 |
| `exp30_vor_or_multimodel_minimax/manifests/vor_or_gate64_rejected_v3.jsonl` | OK | 206 | 909 |
| `exp37_minimax_localdpo_badnoise_rescue/manifests/localdpo_or_train32.jsonl` | OK | 32 | 160 |
| `exp37_minimax_localdpo_badnoise_rescue/manifests/localdpo_or_heldout16.jsonl` | OK | 16 | 80 |
| `exp37_minimax_localdpo_badnoise_rescue/manifests/exp37_badnoise_states.jsonl` | OK | 32 | 128 |
| `exp38_minimax_full_adapter_breakthrough/manifests/localdpo_v2_train30_filtered.jsonl` | OK | 30 | 150 |
| `exp38_minimax_full_adapter_breakthrough/manifests/localdpo_v2_heldout13_filtered.jsonl` | OK | 13 | 65 |
| `exp38_minimax_full_adapter_breakthrough/manifests/badnoise_v2_train30_states.jsonl` | OK | 30 | 120 |

Unique referenced PAI paths: `758`.
Existing paths: `758`.
Missing paths: `0`.
Referenced type counts: `{'dir': 539, 'file': 219}`.
Approximate referenced transfer size from `du -sb`: `2986401326` bytes (2.781 GiB).

The transfer estimate is below the 200GB stop threshold, but it still requires
H20 disk planning because `/home/nvme01` is already around 90% used.

## Required MiniMax Assets

- Exp30 Gate64 V3 train32/heldout16/selected/candidate manifests.
- Exp37 LocalDPO train32/heldout16 and bad-noise states.
- Exp38 LocalDPO v2 train30/heldout13 filtered manifests and bad-noise v2 states.
- Referenced condition/winner/loser/mask frame directories and raw-output mp4s.
- MiniMax official repo and model weights for H20 smoke/debug.

## Assets Not To Transfer

- Full VOR archive.
- Full old logs/autoresearch trees.
- Old failed visual caches not referenced by selected manifests.
- EffectErase VOR full dataset.
- VideoPainter outputs unless a later paper pack explicitly asks for them.

## MiniMax Repo / Weight Readback

- `/mnt/nas/hj/official_repos/MiniMax-Remover_28e12b4`: exists=1, type=dir, size=4096, du=56193453, sha256=
- `/mnt/nas/hj/official_repos/MiniMax-Remover_28e12b4/README.md`: exists=1, type=file, size=4141, du=4141, sha256=44a25c33066b9380510a1a7834f5e12c953af1fa2df3d178982e4fdc0b4812bc
- `/mnt/nas/hj/official_repos/MiniMax-Remover_28e12b4/requirements.txt`: exists=1, type=file, size=271, du=271, sha256=c5751402f92a7624ffe64897e378b83fe3e08d8fec26591c372b216e4eda5ae0
- `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/minimax_remover/current`: exists=1, type=dir, size=108, du=108, sha256=
- `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/minimax_remover`: exists=1, type=dir, size=4096, du=108, sha256=
- `/mnt/nas/hj/H20_Video_inpainting_DPO/weights`: exists=1, type=dir, size=4096, du=1141, sha256=
- `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/README.md`: exists=1, type=file, size=593, du=593, sha256=90e5f26873f4cbc141e50972b338481ea8934678009247a81822362ebf2fcad9

PAI MiniMax model symlink target was read as:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/minimax
```

Target size observed: about `2.6G`, with transformer and VAE safetensors.

## Outputs

- `reports/exp39_pai_minimax_asset_inventory.csv`
- `reports/exp39_pai_minimax_filelist_to_transfer.txt`
- `reports/exp39_pai_minimax_manifest_path_map.json`

## Decision

```text
PAI_MINIMAX_ASSET_INVENTORY_COMPLETED_READ_ONLY
```

Next step is H20 small-file transfer/checksum once H20 SSH is stable enough.
