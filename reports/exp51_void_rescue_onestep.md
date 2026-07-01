# Exp51 VOID One-Step Rescue Grid

Status: `VOID_RESCUE_ONESTEP_BLOCKED`

Exact blocker: `VOID_RESCUE_ONESTEP_BLOCKED_SLOW_FORWARD_NO_CHECKPOINT`

## Attempt

- Recipes requested: R1, R2, R3, R4
- Data requested: VOR-Train train4 / heldout4
- GPU: H20 GPU0
- PID: 3425290
- Runtime before controlled TERM: about 10.5 minutes
- Peak observed VRAM: about 19.9 GB
- Checkpoints created: 0
- Runner reports created: 0
- Heldout videos generated: 0

The process stayed inside the first heavy forward/cache path and produced no checkpoint or report. It was terminated rather than allowed to become an unbounded or long-running training attempt.

## Interpretation

This is a controlled micro-gate blocker, not a negative rescue result. R1/R2/R3/R4 have not been evaluated. 10-step remains locked.

## Next Minimal Fix

Refactor the runner into a narrower R1-only row0 smoke, then expand to train4 recipe-by-recipe after the first checkpoint is observed. Alternatively reuse the proven Exp50 10-step cache path per recipe, rather than keeping all R1-R4 inside one long process.

## Safety

No VOR-Eval, hard comp, official source modification, shared trainer modification, `inference/metrics.py` modification, or 10-step rescue run was performed.
