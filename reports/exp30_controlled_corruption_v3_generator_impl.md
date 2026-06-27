# Exp30 Controlled-Corruption V3 Generator Implementation

Status: `CONTROLLED_CORRUPTION_V3_GENERATOR_IMPLEMENTED`

Date: 2026-06-27

This milestone implements the preregistered controlled-corruption v3 generator
without running Smoke16 v3, Gate64, MiniMax adapter training, or any long
training.

Implemented script:

- `exp30_vor_or_multimodel_minimax/scripts/run_controlled_corruption_smoke16_v3.py`

Locked schedule:

- `CC-v3-B` medium-object profile on all 16 Smoke16 sources.
- `CC-v3-A` mild-object profile on six preregistered repair sources.
- `CC-v3-C` affected-soft profile on two preregistered sources.
- Total controlled candidate cap: 24.
- One deterministic primary controlled candidate per source is selected by
  classification priority, then closeness to the target local defect range,
  temporal stability, and outside preservation.

Safety:

- Existing Smoke16 v2 outputs are not overwritten.
- The script writes a new v3 output tree.
- No VOR-Eval data is used.
- No GPU task was launched by this implementation milestone.
- Left CLI, Exp31, and Exp33 paths/processes were not modified.

Local validation:

- `git diff --check`: passed.
- `python -m py_compile exp30_vor_or_multimodel_minimax/code/*.py exp30_vor_or_multimodel_minimax/scripts/*.py`: passed.
- `python -m unittest discover -s exp30_vor_or_multimodel_minimax/tests -p 'test_*.py'`: passed.
- `bash -n exp30_vor_or_multimodel_minimax/scripts/*.sh`: passed.

Next allowed action:

Run controlled-corruption Smoke16 v3 on the fixed 16-row materialized manifest,
inspect the generated evidence, and only then decide whether the controlled
fallback part is ready for the full Smoke16 v3 aggregate gate.
