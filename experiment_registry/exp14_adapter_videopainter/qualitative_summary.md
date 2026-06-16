# Qualitative Summary

Status: training_completed_visual_eval_blocked.

Gate2000 completed and saved `last_weights`, but no four-column VideoPainter
baseline vs adapter visualization exists yet.

Required visualization columns:

1. GT
2. mask overlay
3. VideoPainter baseline
4. VideoPainter + DPO adapter

The visualization should be produced only after an Exp14-specific DAVIS eval
adapter is implemented. Do not reuse upstream VideoPainter eval outputs as the
final project visualization unless they are adapted to the fixed raw6 hard-comp
protocol and existing project metric backend.
