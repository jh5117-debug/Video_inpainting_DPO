# Exp35 Status

Current status: `EXP35_READBACK_COMPLETED`

## 2026-06-27 Readback

- Branch: `research/exp35-minimax-flow-dpo-rescue-20260627`.
- Base HEAD: `f69688fe4ff96c4d4f0dcd308eef69822fc1035b`.
- Exp30 Gate64 pool: `VOR_OR_GATE64_MULTIMODEL_POOL_READY`.
- Exp30 MiniMax gate: `MINIMAX_ADAPTER_RECIPE_NOT_READY`.
- Failure class at readback: recipe/update-state suspected, not data
  availability or basic plumbing.
- Protected lanes checked read-only.
- No GPU inference, training, 30-step, RC-FPO, or protected-lane action
  launched.

Report:

- `reports/exp35_minimax_rescue_readback.md`

## 2026-06-27 No-Change Forensic Audit

- Status: `MINIMAX_NOCHANGE_CAUSE_UTILITY_SCALE_TOO_WEAK`.
- Training launched: false.
- Checkpoint keys: 461 common keys and 0 missing/unexpected keys for both
  frozen and EMA recipes.
- Frozen parameter delta/param norm ratio: `5.6404525516172905e-06`.
- EMA parameter delta/param norm ratio: `5.630459939756668e-06`.
- Step0/Step10 byte-identical rows: 0/32.
- Mean full/mask/affected/outside abs pixel diff:
  `0.13143352206508793`, `0.18672874342540607`,
  `0.1731182035360047`, `0.10850902535158265`.
- Frozen linear utility mean/min/max:
  `0.4999982982873917` / `0.49997058510780334` /
  `0.5000085830688477`.
- EMA linear utility mean/min/max:
  `0.5000003516674042` / `0.49999284744262695` /
  `0.5000050663948059`.
- Decision: run inference-sensitivity positive-control before recipe redesign.
