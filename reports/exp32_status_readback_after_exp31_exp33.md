# Exp32 Status Readback After Exp31 / Exp33

Date: 2026-06-28

Status: `EXP32_READBACK_REFRESHED_DATA_GATE_STILL_PENDING`

## Git

- branch: `research/exp32-diffueraser-vor-or-2000step-20260627`
- local HEAD at readback: `f65e5286a02f3d744882cd1c93071cb4e7085c84`
- remote HEAD at readback: `f65e5286a02f3d744882cd1c93071cb4e7085c84`
- latest commit: `Refresh Exp32 readback status`

## Decision

Exp32 remains readback-only in this prompt. No DiffuEraser VOR-OR training,
candidate generation, loser mining, adapter run, or GPU study was launched.

The current blocker remains the data gate: Exp32 still needs an explicitly
authorized train32 plus heldout16 scene-disjoint VOR-OR data gate before any
Stage1/Stage2 2000-step training can be considered.

Exp30 Gate64 or other historical pools must not be reused for Exp32 unless the
user separately authorizes that data decision. This report does not authorize
VOR-OR training.
