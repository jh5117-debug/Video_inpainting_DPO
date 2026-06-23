# Exp27 LocalDPO Compatibility Patch

Date: 2026-06-23

File added:

`exp27_paper_grounded_preference_study/code/localdpo_compat.py`

Tests updated:

`exp27_paper_grounded_preference_study/tests/test_official_parity_helpers.py`

## Patch Contract

The official LocalDPO repository is read-only. The Exp27 wrapper:

- imports the official module from the cached official commit;
- does not edit official source files;
- replaces only the matplotlib canvas conversion inside `get_random_shape`;
- keeps the official mask motion logic unchanged;
- records the patch as `OFFICIAL_CODE_COMPATIBILITY_PATCH`.

## Validation

Command:

```bash
python -m unittest exp27_paper_grounded_preference_study.tests.test_official_parity_helpers -v
```

Result:

5 tests passed.

Covered:

- official LocalDPO raw path remains recorded as blocked by reshape;
- compat LocalDPO path is deterministic and returns expected mask shape;
- SDPO toy safe-lambda remains passing;
- Linear-DPO toy utility/EMA remains passing.

## Limitation

This resolves only the random moving mask generator runtime compatibility. It does not yet validate LocalDPO progressive corruption, outside latent reinjection/fusion, or real DiffuEraser-batch DPO parity.
