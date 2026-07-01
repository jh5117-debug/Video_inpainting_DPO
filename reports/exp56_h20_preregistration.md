# Exp56-H20 R5 Preregistration

Status: `EXP56_H20_R5_PREREGISTERED`

This H20 lane runs only two cells:

1. `R5_Q2_T500_S0`
2. `R5_HALF_Q2_T500_S0`

Common settings:

- Quadmask: Q2 strict affected
- Timestep: T500
- Scope: S0 `proj_out`
- Data: train4 / heldout4 from VOR-Train only
- Steps: one optimizer step only
- 10-step: not allowed
- VOR-Eval: not used
- Hard comp: not used

R5 objective:

- Object-only DPO
- No affected DPO push
- Affected / overlap / boundary are preservation penalties
- `loser_grad_scale = 0.0`
- `winner_anchor = 0.10`
- `outside_preservation = 0.10`
- `boundary_preservation = 0.15`
- `affected_preservation = 0.10`
- `overlap_preservation = 0.15`
- Local-only margin

R5_HALF:

Same objective as R5 with a reduced update. The selected implementation is half learning rate: `lr = 5e-6`, leaving optimizer semantics unchanged and avoiding post-hoc parameter scaling.

Expected target behavior:

- Object/mask not worse
- Overlap, affected, and boundary safe under the stricter Exp56 gate
- Outside safe
- Visual tie-or-better
- Loser contribution near zero

No cell was run before this preregistration.
