# Exp51 VOID Quadmask Ablation Data

Status: `VOID_QUADMASK_ABLATION_READY`

Built Q0/Q1/Q2/Q3 quadmask variants for existing VOR-Train train4/heldout4 rows only. VOR-Eval is excluded and no hard comp was used.

## Variants

- Q0 current: current Exp50 quadmask copied as-is.
- Q1 object-only: object/overlap -> 0, everything else -> 255.
- Q2 strict affected: high-threshold affected map from abs(V_obj - V_bg), object preserved.
- Q3 broad affected: lower-threshold affected map from abs(V_obj - V_bg), object preserved.

## Validation

Every generated row decodes, contains non-background/object structure, records area statistics, and has an evidence sheet for visual review.


## Codex Visual Review

All 8 sample evidence sheets were opened in a local montage. Q1 object-only is clean and avoids affected-region spill. Q2 strict affected is the most conservative local affected candidate. Q3 broad affected is diagnostically useful but visibly broad on several REAL rows, so it should not be promoted before one-step safety checks.
