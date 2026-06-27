# Exp36 Metric Summary

No Exp36 metrics have been generated yet.

Readback imported prior findings:

- Exp30 MiniMax 10-step: heldout visual better `0/32`.
- Exp35 R1/R2/R3 rescue 10-step: heldout visual better `0/48`.
- Exp35 R1/R2/R3 mean mask PSNR deltas were all negative.

## 2026-06-27 No-Change Forensic Audit

Prior metrics summarized under Exp36:

- Exp30 frozen/EMA utility means: `0.4999982983` and `0.5000003517`.
- Exp30 frozen/EMA delta/param-norm ratios: `5.64045e-06` and
  `5.63046e-06`.
- Exp35 R1/R2/R3 mean mask PSNR deltas: `-0.048611`, `-0.053910`,
  `-0.081454`.
- Exp35 R1/R2/R3 mean boundary PSNR deltas: `-0.423993`, `-0.434234`,
  `-0.493050`.

## 2026-06-27 Inference Sensitivity Test

- Identity replay max full MAE: `0.0`.
- Perturbation mean full MAE: `0.08821829589193357`.
- Perturbation mean mask MAE: `0.15630244233590715`.
- Perturbed tensors: `16`.
- Perturb scale: `1.01`.

These are sensitivity metrics, not quality metrics.

## 2026-06-27 Trainable Scope Audit

No model-quality metrics were generated. The scope contract records:

- S1 LoRA rank: `8`.
- S1 LoRA alpha: `16`.
- S1 dropout: `0.0`.
- S2 status: locked until S1 positive-control evidence.

This milestone adds implementation and checkpoint-roundtrip tests only; it
does not change MiniMax quality status.
