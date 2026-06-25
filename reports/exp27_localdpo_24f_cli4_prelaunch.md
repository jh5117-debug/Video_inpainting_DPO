# Exp27 LocalDPO 24F CLI4 Prelaunch

Date: 2026-06-25

Status: `LOCALDPO_24F_CLI4_READY_NOT_LAUNCHED`

Branch: `research/exp27-localdpo-objective-cli4-20260625`

## Readback

Latest Exp27 source-of-truth shows:

- `TRUE_MODEL_PARITY`
- `SDPO_TRUE_MODEL_32X4_SCAN_COMPLETE`
- `SDPO_TINY_STEP_ACTUAL_CHECK_PASSED`
- `LINEAR_TRUE_MODEL_1_10_STEP_PASSED`
- `LOCALDPO_24F_PENDING`
- `RCFPO_NOT_STARTED`

Therefore CLI4 claims only the next pending milestone: LocalDPO DiffuEraser
24F adaptation.

## Runner

Added:

- `exp27_paper_grounded_preference_study/code/localdpo_24f_adaptation.py`
- `exp27_paper_grounded_preference_study/scripts/run_exp27_localdpo_24f_adaptation.py`
- `exp27_paper_grounded_preference_study/tests/test_localdpo_24f_adaptation.py`

The runner stages:

1. `P8`: real clean winner + official LocalDPO moving mask + SFT DiffuEraser
   self-model loser + outside reinjection/composite.
2. `P32`: same protocol at 32 pairs.
3. `objective`: only after P32 passes, run original LocalDPO-style
   `RA-DPO + global DPO + SFT` 1-step and 10-step.

## Guardrails

- Controlled corruption preview is test-only and forced non-gate-valid.
- P32 cannot pass without `32/32` review artifacts.
- Objective cannot run before P32 unless an explicit debugging override is
  passed.
- No 50-step, no four-grid 50-step, no O0-O5, and no RC-FPO are launched by
  this runner.
- No shared trainer and no `inference/metrics.py` changes.

## CLI4 Output Roots

- logs:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp27_paper_grounded_preference_study/cli4/`
- experiments:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp27_paper_grounded_preference_study/cli4/`

## Expected Launch

GPU2 only, after the required GPU1 stagger and GPU0 read-only health check.
