# Exp35 Qualitative Summary

No new Exp35 videos have been generated or reviewed yet.

Readback imports the Exp30 finding: Codex reviewed Step0/Step10 MiniMax
heldout evidence pages and found Step10 visually tied with Step0 in all
recipe-row comparisons. Exp35 must not write any `PASS`, `QUALITY_POSITIVE`,
or `THIRD_BACKBONE` language until new videos are generated and inspected.

## 2026-06-27 No-Change Forensic Audit

No new videos were generated. The audit used existing Exp30 heldout frames and
confirmed that Step10 outputs are not byte-identical to Step0, but the mean
pixel movement is sub-perceptual. This supports the prior visual review:
Step10 was a tie/no visible improvement rather than a hidden positive.

## 2026-06-27 Inference Sensitivity Positive-Control

Codex opened all 4 generated temporal comparison strips. The Step0 identity
control was visually identical and hash-identical on all rows. The temporary
perturbed checkpoint produced subtle nonzero texture/noise-level changes on
all rows, with no visual collapse, black/purple artifact, color drift,
systematic outside damage, or obvious new temporal artifact.

Qualitative interpretation: inference does not ignore MiniMax transformer
weights, but the output is low-sensitivity to small weight movement. This
supports continuing to trainable-scope and objective-scale diagnostics before
any rescue recipe training.

## 2026-06-27 Trainable-Scope Audit

No new videos were generated. The audit ties the previous visual sensitivity
result to the trainable scope: Exp30 used the full MiniMax transformer and the
temporary perturbation was visible as subtle nonzero response. The lack of
quality-positive Step10 movement should not be described as a LoRA/scope
visibility failure.
