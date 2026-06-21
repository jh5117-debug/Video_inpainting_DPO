# Exp23 Pair001 Boundary Mode Audit

Date: 2026-06-21

Pair:

```text
pair_id = phaseA_scale1_pair001_outer2_gpus2456
fresh control = fresh_exp11_outer_b075
candidate = candidate_scale1_outer2_b075
```

## Verdict

```text
PAIR001_CONTROL_INVALID_BOUNDARY_MODE
```

The completed fresh Exp11 control cannot be used as an `outer` Exp11 control.

## Runtime Evidence

Reliable runtime evidence comes from each stage's `dpo_diagnostics.csv`, which
records the effective `boundary_mode` used by the loss map.

| model | stage | boundary_mode | mask weight | boundary weight | outside weight | verdict |
|---|---|---|---:|---:|---:|---|
| fresh_exp11_outer_b075 | Stage1 | both | 1.0 | 0.75 | 0.05 | invalid |
| fresh_exp11_outer_b075 | Stage2 | both | 1.0 | 0.75 | 0.05 | invalid |
| candidate_scale1_outer2_b075 | Stage1 | exp23_pool_morphology | 1.0 | 0.75 | 0.05 | not control |
| candidate_scale1_outer2_b075 | Stage2 | exp23_pool_morphology | 1.0 | 0.75 | 0.05 | not control |

Evidence paths:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp23_two_stage_pool_morphology_sweep/pairs/phaseA_scale1_pair001_outer2_gpus2456/fresh_exp11_outer_b075/stage1/dpo_diagnostics.csv
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp23_two_stage_pool_morphology_sweep/pairs/phaseA_scale1_pair001_outer2_gpus2456/fresh_exp11_outer_b075/stage2/dpo_diagnostics.csv
```

## Cause

The isolated Exp23 legacy map path called `build_region_loss_weight_map()`.
That helper used:

```python
os.environ.get("BOUNDARY_MODE", "both")
```

The runner did not explicitly pass `--boundary_mode outer`, and the stage logs
only showed `--legacy_exact true --outer_pool_steps 1`, which is not sufficient
runtime evidence for `outer` because the legacy path ignored those morphology
fields and used the environment/default boundary mode.

## Decision

Do not run scientific DAVIS50 comparison on this completed pair.

Required action:

1. Make Exp23 boundary mode explicit in Stage1, Stage2, and runner.
2. Refuse missing `--boundary_mode` rather than silently using `both`.
3. Rerun the pair from scratch with:
   - fresh Exp11: `legacy_exact=true`, `boundary_mode=outer`, `outer_pool_steps=1`
   - candidate: `legacy_exact=false`, `boundary_mode=outer`, `outer_pool_steps=2`
4. Run paired DAVIS50 only after corrected training completes.

