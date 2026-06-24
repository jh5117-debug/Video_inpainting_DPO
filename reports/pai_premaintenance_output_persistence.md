# PAI Pre-Maintenance Output Persistence

Status: `BLOCKED_NAS_PERMISSION`

Date: 2026-06-25

This milestone was attempted before launching any new Exp25/Exp26/Exp27 work,
because PAI may enter maintenance and the large non-Git artifacts under
`/home/hj` need durable storage.

## Source Artifacts

| Experiment | Source | Files | Bytes |
| --- | --- | ---: | ---: |
| Exp25 | `/home/hj/exp25_gate32_dense_review_runs` | 99 | 66982608 |
| Exp26 | `/home/hj/exp26_gate64_runs/gate64_official_43597cf_20260625_031155` | 14408 | 8405904095 |

## Intended NAS Targets

| Experiment | Intended Destination |
| --- | --- |
| Exp25 | `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_vor_or_preference_data/gate32_dense_review_20260625` |
| Exp26 | `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/gate64_official_43597cf_20260625_031155` |

## Blocker

The SSH user `hj` can read the artifacts in `/home/hj`, but cannot create
directories or write files under the requested NAS project root:

- `/mnt/nas/hj/H20_Video_inpainting_DPO`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/logs`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/reports`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/data`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments`

Additional checks:

- `sudo -n true`: failed, password required.
- `ssh root@47.103.26.60` with the current key: failed.
- shallow `find -writable` under the NAS project root found no writable
  destination.

Because this milestone requires preserving large non-Git artifacts to durable
NAS storage before starting new experiments, no new Exp25 root-cause matrix,
Exp26 Gate64 review/source repair, VideoPainter DPO, or Exp27 GPU task was
launched after this blocker was detected.

## Completion Markers

Not created:

- `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/EXP25_GATE32_REVIEW_PERSISTED_TO_NAS`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/EXP26_GATE64_PERSISTED_TO_NAS`

The source directories remain intact on PAI `/home/hj`.

