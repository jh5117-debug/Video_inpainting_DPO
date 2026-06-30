# Exp50 Metric Summary

Last updated: 2026-06-30T16:44:02+08:00

No training or inference metrics yet.

Weight relay:
- VOID files: 12
- Base files: 40
- SHA match: yes

Environment smoke:
- Status: `VOID_ENV_PARTIAL`
- Imports pass: 44
- CUDA failures: 0
- Version pin warnings: 8

VOR-to-VOID Gate8:
- Status: `VOID_VOR_QUADMASK_GATE8_READY`
- Gate rows: 8
- REAL / BLENDER: 4 / 4
- Small / medium / large: 3 / 3 / 2
- Scene overlap: False
- VOR-Eval excluded: True

## C2 Environment Repair (2026-06-30T15:25:36+08:00)

No inference metrics were produced. `VOID_ENV_READY` was not reached due to `VOID_ENV_BLOCKED_TORCH` and `VOID_ENV_BLOCKED_DEEPSPEED`.

## C3 Environment Relay Ingest (2026-06-30T16:44:02+08:00)

- Status: `VOID_ENV_READY`
- Wheelhouse files: 145 (3.3G)
- Transfer hash match: True; missing 0; mismatches 0
- Env torch: `2.7.1+cu126`; CUDA runtime `12.6`
- Import failures: 0
- CUDA bf16 tiny smoke: `PENDING_NO_FREE_GPU`; max allocation 67146240 bytes
