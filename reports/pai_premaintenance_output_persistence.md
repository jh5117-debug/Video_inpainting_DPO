# PAI Pre-Maintenance Output Persistence

Status: `PERSISTENCE_PASSED`

Date: 2026-06-25

This persistence gate was attempted before launching any new Exp25/Exp26/Exp27
work. The artifacts requiring immediate durable persistence are Exp25 Gate32
dense review outputs and Exp26 Gate64 official generation outputs under PAI
`/home/hj`.

The SSH user `hj` cannot write under:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch`

`sudo -n` is not available and root SSH with the current key failed. Therefore
the required NAS persistence markers were not created, and no new Exp27
true-model forward parity GPU task was launched after this blocker.

The current Exp27 state remains:

- `TRUE_MODEL_FORWARD_READBACK_COMPLETE`
- `SDPO_REAL_RESIDUAL_PROXY_ONLY`
- `OBJECTIVE_STUDY_PENDING`
- `RCFPO_NOT_STARTED`

## Resolution

Status: `PERSISTENCE_PASSED`

The PAI WebIDE root session granted `hj` write access to the required NAS
runtime and autoresearch directories. HAL then verified SSH-key login as `hj`
and write access to:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch`

The required cross-track `/home` artifacts were persisted to NAS without
starting any new Exp27 GPU task.

Verified copy:

| Experiment | Source files | Destination files | Source bytes | Destination bytes | Inventory | SHA256 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Exp25 | 99 | 99 | 66982608 | 66982608 | OK | OK |
| Exp26 | 14408 | 14408 | 8405904095 | 8405904095 | OK | OK |

Runtime markers now exist:

- `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/EXP25_GATE32_REVIEW_PERSISTED_TO_NAS`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/EXP26_GATE64_PERSISTED_TO_NAS`

PAI summary and checksum artifacts:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/premaintenance_persistence_20260625/`
