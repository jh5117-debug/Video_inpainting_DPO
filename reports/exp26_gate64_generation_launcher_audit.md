# Exp26 Gate64 Generation Launcher Audit

Status: `IMPLEMENTED_PENDING_PAI_RUN`

## What Changed

Added isolated Gate64 generation plumbing inside `exp26_videopainter_dpo_v2/`:

- `code/extract_vp2_gate64_vor_bg.py`
- `code/run_vp2_gate64_official_generation.py`
- `scripts/run_vp2_gate64_generation_pai.sh`

Updated:

- `code/generate_vp2_moving_br_masks.py`
- `tests/test_vp2_49f_materialization_and_masks.py`

## Correctness Fix

`vp2_mixed_br_mask_v1` requires six mask families:

- irregular free-form
- object-like polygon
- soft blob
- edge-touch free-form
- ellipse/circle subset
- thin-structure free-form

The generator now reads the locked per-row fields:

- `mask_profile`
- `area_bucket`
- `motion_bucket`
- `deformation_bucket`
- `edge_touch_target`

and records actual generated-mask metadata in `mask_meta`.

## PAI Pipeline

The PAI launcher performs, in order:

1. Exact VOR-Train/BG selective extraction from split archive parts.
2. Formal first-49-unique-frame materialization.
3. Locked mixed moving-mask generation with first frame GT.
4. Official VideoPainter 49F generation.

It writes:

- raw frames and raw mp4
- diagnostic hard-comp frames and mp4
- side-by-side mp4
- contact sheets
- `gate64_generation_status.csv`
- `gate64_generation_summary.json`

Diagnostic comp is saved for review only. It is not treated as the primary
loser unless later data-readiness explicitly selects a raw/comp protocol.

## GPU Policy

The launcher selects the first free GPU in `0..6` if `CUDA_VISIBLE_DEVICES` is
unset. GPU7 is excluded.

## Validation

Commands run before this report:

```bash
python -m py_compile exp26_videopainter_dpo_v2/code/*.py
python -m unittest discover -s exp26_videopainter_dpo_v2/tests -p 'test_*.py'
bash -n exp26_videopainter_dpo_v2/scripts/*.sh
git diff --check
```

Result:

- py_compile: passed
- unit tests: `23` passed
- bash syntax: passed
- diff check: passed

## Next

Commit and push this implementation, create a versioned PAI snapshot from that
commit, then launch Gate64 generation from the locked manifest.
