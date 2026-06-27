# Exp33 EffectErase VOR-Eval Official81 Runner Scaffold

Status: `EXP33_VOREVAL_OFFICIAL81_RUNNER_READY`

Exp33 now has a dedicated runner for the held-out VOR-Eval official81
EffectErase baseline:

`exp33_effecterase_vor_eval_baseline/scripts/run_effecterase_vor_eval_official81.py`

## Scope

- Runs official EffectErase inference only.
- Uses the Exp33 materialized VOR-Eval official81 manifest.
- Writes raw EffectErase outputs as primary artifacts.
- Writes command validation, per-row status, and inference summary reports.
- Does not launch adapter training, loser mining, DPO, zero-gap, one-step, or
  checkpoint selection.

## Guardrails

The runner blocks rows unless all of the following are true:

- `vor_eval=true`
- `eligible_for_training=false`
- `source_role=held_out_vor_eval_baseline`
- `scientific_role=held_out_baseline_only_not_training`
- `raw_output_primary=true`
- `output_path` is under the Exp33 output root

## Validation

Local validation passed:

- `python3 -m py_compile exp33_effecterase_vor_eval_baseline/scripts/run_effecterase_vor_eval_official81.py`
- `python3 -m py_compile exp33_effecterase_vor_eval_baseline/tests/test_vor_eval_official81_audit.py`
- `python3 -m unittest exp33_effecterase_vor_eval_baseline.tests.test_vor_eval_official81_audit`

Command validation against the PAI EffectErase venv/assets is still pending.
EffectErase inference has not started in this scaffold milestone.
