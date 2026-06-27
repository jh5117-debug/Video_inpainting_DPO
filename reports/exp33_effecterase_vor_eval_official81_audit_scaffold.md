# Exp33 EffectErase VOR-Eval Official81 Audit Scaffold

Status: `EXP33_VOREVAL_OFFICIAL81_AUDIT_READY`

The Exp33 branch now includes a dedicated compatibility audit for held-out
VOR-Eval official 81-frame EffectErase baseline inputs.

It differs from Exp29 diagnostic smoke scripts in one important way:
`vor_eval=true` is allowed, but only with `eligible_for_training=false` and
`scientific_role=held_out_baseline_only_not_training`.

Validation:

- `git diff --check`: passed
- `py_compile`: passed
- Exp33 audit unit tests: 3 passed
- Exp29 scaffold unit tests: 2 passed
- `bash -n`: passed

No EffectErase inference, adapter training, loser mining, or checkpoint
selection was launched by this scaffold.
