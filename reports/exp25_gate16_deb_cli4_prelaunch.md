# Exp25 Gate16 DE-B CLI4 Prelaunch

Date: 2026-06-25

Status: `GATE16_DEB_CLI4_READY_NOT_LAUNCHED`

Purpose:

- run the fixed DE-B Gate16 confirmation after the root-cause matrix selected `DE-B_sft_raw6_d8_propainter`;
- avoid repeating Gate32 or the 12-sample root-cause matrix;
- keep outputs isolated under `cli4` paths.

Added code:

- `exp25_vor_or_preference_data/scripts/select_gate16_deb_sources.py`
- `exp25_vor_or_preference_data/scripts/launch_exp25_gate16_deb.py`
- `exp25_vor_or_preference_data/tests/test_gate16_deb_selection.py`

Validation passed:

```text
python -m py_compile exp25_vor_or_preference_data/scripts/select_gate16_deb_sources.py exp25_vor_or_preference_data/scripts/launch_exp25_gate16_deb.py
python -m unittest exp25_vor_or_preference_data.tests.test_gate16_deb_selection
bash -n exp25_vor_or_preference_data/scripts/*.sh
git diff --check
```

No Gate16 output or scientific result exists yet.
