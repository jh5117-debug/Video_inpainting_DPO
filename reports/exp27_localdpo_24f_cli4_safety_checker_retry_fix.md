# Exp27 CLI4 LocalDPO 24F Safety-Checker Retry Fix

Date: 2026-06-25

Branch: `research/exp27-localdpo-objective-cli4-20260625`

## Context

CLI4 launched the Exp27 LocalDPO DiffuEraser 24F adaptation on GPU2 after
readback confirmed:

- true SDPO policy/reference parity is complete;
- true Linear-DPO Frozen/EMA 1/10-step gate is complete;
- `LOCALDPO_24F_PENDING`;
- `RCFPO_NOT_STARTED`.

## Failure

The first P8 run failed during DiffuEraser self-model loser generation because
`diffueraser/diffueraser_OR.py` attempted to load:

`/mnt/nas/hj/weights/stable-diffusion-v1-5/safety_checker/config.json`

That component is absent from the local PAI weight mirror. The related
non-OR DiffuEraser path already disables the safety checker explicitly.

## Fix

The isolated Exp27 CLI4 branch now passes these arguments to
`StableDiffusionDiffuEraserPipeline.from_pretrained` in
`diffueraser/diffueraser_OR.py`:

- `safety_checker=None`
- `feature_extractor=None`
- `requires_safety_checker=False`

This keeps the inference path aligned with the existing local DiffuEraser
behavior and avoids depending on an unavailable research-weight component.

## Verification

- `python -m py_compile diffueraser/diffueraser_OR.py exp27_paper_grounded_preference_study/code/localdpo_24f_adaptation.py exp27_paper_grounded_preference_study/scripts/run_exp27_localdpo_24f_adaptation.py`
- `python -m unittest exp27_paper_grounded_preference_study.tests.test_localdpo_24f_adaptation exp27_paper_grounded_preference_study.tests.test_localdpo_full_adapter exp27_paper_grounded_preference_study.tests.test_official_parity_helpers`
- `git diff --check`

## Guardrail Status

- P8: pending retry.
- P32: not started.
- LocalDPO original objective 1/10-step: not started.
- O0-O5: not started.
- 50-step: not started.
- RC-FPO: `NOT_STARTED`.
