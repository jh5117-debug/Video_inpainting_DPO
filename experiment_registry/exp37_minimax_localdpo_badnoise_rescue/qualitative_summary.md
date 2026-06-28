# Exp37 Qualitative Summary

No Exp37 videos have been generated yet.

Readback imported prior MiniMax qualitative status:

- Exp30 visual better rows: `0/32`.
- Exp35 visual better rows: `0/48`.
- Exp36 visual better rows: `0/24`.

The next qualitative question is whether train-side videos improve while
heldout does not, or whether both train and heldout remain non-positive.

## 2026-06-28 Train-vs-Heldout Diagnosis

Codex reviewed `32/32` Step0-vs-Step10 temporal strips:

- Train16: `0/16` better, `16/16` tie/no visible change, `0` new artifacts.
- Heldout16: `0/16` better, `16/16` tie/no visible change, `0` new artifacts.

Representative checked rows included the best and worst metric rows:
`BLENDER_GRASS001_00001`, `BLENDER_MOUNTAIN008_00001`,
`REAL_ENV087_00002_001_01`, and `REAL_ENV105_00004_001_01`. Step10 was
visually indistinguishable from Step0 in all checked strips; the diff columns
were near black.

Qualitative diagnosis: `MINIMAX_OBJECTIVE_SIGNAL_TOO_WEAK`.
