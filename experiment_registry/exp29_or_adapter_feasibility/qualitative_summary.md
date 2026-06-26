# Exp29 Qualitative Summary

No Exp29 video evidence has been generated yet. Any future visual pass must be
based on per-sample video evidence, not a run-level montage alone.

MiniMax is the only audited model currently eligible for a future Exp29 visual
smoke. EffectErase cannot produce visual evidence until the official LoRA and
Wan base assets are installed.

## 2026-06-26 MiniMax Inference Smoke

- `davis_bear`: medium-hard candidate; object mostly removed, mild smoothing.
- `davis_bus`: trivial-bad; bus remains.
- `davis_mallard-water`: trivial-bad; duck remains with blue/black artifact.
- `davis_elephant`: trivial-bad; elephant remains with haze/smoothing.

MiniMax inference is technically runnable but not yet a robust OR baseline.
EffectErase visual review is blocked because official weights are missing.

## 2026-06-26 MiniMax 10-Step Heldout Review

- `davis_hockey`: Step10 and Step0 are visually tied; target remains.
- `davis_koala`: Step10 and Step0 are visually tied; fur texture remains
  smoothed/residual.

No heldout visual improvement was found. This is a technical adapter-plumbing
pass, not a third-backbone scientific positive.
