# PAI Pre-Maintenance Output Persistence

Status: `PERSISTENCE_PASSED`

Date: 2026-06-25

This milestone was attempted before launching any new Exp25/Exp26/Exp27 work,
because PAI may enter maintenance and the large non-Git artifacts under
`/home/hj` need durable storage.

## Source Artifacts

| Experiment | Source | Files | Bytes |
| --- | --- | ---: | ---: |
| Exp26 | `/home/hj/exp26_gate64_runs/gate64_official_43597cf_20260625_031155` | 14408 | 8405904095 |
| Exp25 | `/home/hj/exp25_gate32_dense_review_runs` | 99 | 66982608 |

## Intended NAS Targets

| Experiment | Intended Destination |
| --- | --- |
| Exp26 | `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/gate64_official_43597cf_20260625_031155` |
| Exp25 | `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_vor_or_preference_data/gate32_dense_review_20260625` |

## Blocker

The SSH user `hj` can read the artifacts in `/home/hj`, but cannot create
directories or write files under the requested NAS project root:

- `/mnt/nas/hj/H20_Video_inpainting_DPO`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/logs`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/reports`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/data`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments`

Observed ownership/permission pattern: the project NAS root and key
subdirectories are owned by `root:root` or another uid and are not writable by
`hj`.

Additional checks:

- `sudo -n true`: failed, password required.
- `ssh root@47.103.26.60` with the current key: failed.
- shallow `find -writable` under the NAS project root found no writable
  destination.

Because this milestone requires preserving large non-Git artifacts to durable
NAS storage before starting new experiments, no new Exp26 Gate64 review, source
repair, VideoPainter DPO, Exp25 root-cause matrix, or Exp27 GPU task was
launched after this blocker was detected.

## Completion Markers

Not created:

- `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/EXP26_GATE64_PERSISTED_TO_NAS`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/EXP25_GATE32_REVIEW_PERSISTED_TO_NAS`

## Required Fix

Run the persistence command from a user/session that can write to the NAS
project root, or grant `hj` write permission to:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch`

The source directories remain intact on PAI `/home/hj` at the paths listed
above.

## Resolution

Status: `PERSISTENCE_PASSED`

The PAI WebIDE root session granted `hj` write access to the required NAS
runtime and autoresearch directories. HAL then verified SSH-key login as `hj`
and write access to:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch`

The existing PAI `/home` artifacts were persisted to NAS without starting any
new experiment or GPU task.

## Verified NAS Copy

| Experiment | Source files | Destination files | Source bytes | Destination bytes | Inventory | SHA256 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Exp26 | 14408 | 14408 | 8405904095 | 8405904095 | OK | OK |
| Exp25 | 99 | 99 | 66982608 | 66982608 | OK | OK |

Targets:

- Exp26:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/gate64_official_43597cf_20260625_031155`
- Exp25:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_vor_or_preference_data/gate32_dense_review_20260625`

Runtime markers now exist:

- `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/EXP26_GATE64_PERSISTED_TO_NAS`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/EXP25_GATE32_REVIEW_PERSISTED_TO_NAS`

PAI summary and checksum artifacts:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/premaintenance_persistence_20260625/`
