# PAI Pre-Maintenance Output Persistence

Status: `BLOCKED_NAS_PERMISSION`

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

