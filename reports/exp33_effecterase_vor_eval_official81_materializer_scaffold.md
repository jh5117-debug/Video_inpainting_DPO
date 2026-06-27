# Exp33 EffectErase VOR-Eval Official81 Materializer Scaffold

Status: `EXP33_VOREVAL_OFFICIAL81_MATERIALIZER_READY`

The Exp33 branch now includes a dedicated materializer for held-out VOR-Eval
official81 EffectErase baseline inputs.

Guardrails:

- requires `vor_eval=true`;
- requires `eligible_for_training=false`;
- requires `source_role=held_out_vor_eval_baseline`;
- writes only Exp33 output paths;
- launches no EffectErase inference.

Validation:

- `git diff --check`: passed
- `py_compile`: passed
- Exp33 audit/materializer unit tests: 4 passed
- Exp29 scaffold unit tests: 2 passed
- `bash -n`: passed
