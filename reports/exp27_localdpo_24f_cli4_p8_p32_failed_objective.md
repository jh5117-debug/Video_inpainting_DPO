# Exp27 CLI4 LocalDPO 24F P8/P32 Result and Failed Objective

Date: 2026-06-25

Branch: `research/exp27-localdpo-objective-cli4-20260625`

Runtime snapshot:

`/home/hj/runtime_code_snapshots/cli4_exp27_f1aa52d57a7a`

Output root:

`/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp27_paper_grounded_preference_study/cli4/localdpo24f_cli4_retry1_20260625_190006`

## Gate Results

P8:

- status: `P8_PASSED`
- pairs: `8`
- technical valid: `8`
- medium-hard: `2`
- hard-plausible: `6`
- trivial bad: `0`
- technical invalid: `0`
- global collapse: `0`
- outside preservation: passed
- video review: `8/8`

P32:

- status: `P32_PASSED`
- pairs: `32`
- technical valid: `32`
- medium-hard: `14`
- hard-plausible: `17`
- trivial bad: `1`
- technical invalid: `0`
- global collapse: `0`
- outside preservation: passed
- video review: `32/32`

P32 manifest:

`/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp27_paper_grounded_preference_study/cli4/localdpo24f_cli4_retry1_20260625_190006/pairs/p32/exp27_localdpo_24f_p32_manifest.jsonl`

## Objective Failure

The runner attempted to continue into the original LocalDPO objective after
P32 passed, but failed before producing the 1-step/10-step objective summary:

```text
AttributeError: 'Namespace' object has no attribute 'manifest'
```

The failure occurred in:

`exp27_paper_grounded_preference_study/scripts/run_exp27_localdpo_24f_adaptation.py`

The objective summary file was not generated.

## Decision

Status:

`LOCALDPO_ORIGINAL_OBJECTIVE_FAILED_FINAL`

This is the second Exp27 CLI4 lane failure after the safety-checker retry fix.
CLI4 therefore does not apply a second runtime fix or resume this lane again.

## Guardrails

- P8/P32 pair generation is a technical pass.
- The LocalDPO original objective baseline is not complete.
- No 50-step run was started.
- No O0-O5 objective study was started.
- RC-FPO remains `NOT_STARTED`.
