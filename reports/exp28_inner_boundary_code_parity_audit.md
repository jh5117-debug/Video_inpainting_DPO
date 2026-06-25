# Exp28 Inner Boundary Code Parity Audit

Date: 2026-06-25

Status: `CODE_TEST_READY_NOT_LAUNCHED`

Checks:

- radius 0 reproduces the fresh Exp11 outer-control weight map within tolerance.
- outer ring is unchanged for main radii 2, 4, and 8 px.
- regions are non-negative and sum to one.
- inner radius is implemented by image-space pixel erosion.
- inner and outer regions do not illegally overlap.
- Stage1 and Stage2 receive the same explicit Exp28 geometry.
- world_size=2 uses gradient accumulation 2, preserving effective global batch 4 from the old four-card runner.

Validation commands passed:

```text
python -m unittest discover -s exp28_fine_inner_boundary_sweep/tests -p 'test_*.py'
python -m py_compile exp28_fine_inner_boundary_sweep/code/inner_boundary_geometry.py exp28_fine_inner_boundary_sweep/code/exp28_trial_runner.py exp28_fine_inner_boundary_sweep/code/train_exp28_stage1.py exp28_fine_inner_boundary_sweep/code/train_exp28_stage2.py exp28_fine_inner_boundary_sweep/code/summarize_exp28_pair_eval.py
bash -n exp28_fine_inner_boundary_sweep/scripts/eval_exp28_pair_davis50_pai.sh
```

No scientific result is claimed.
