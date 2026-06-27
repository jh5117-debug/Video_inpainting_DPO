# Exp35 Metric Summary

No new Exp35 metrics have been generated yet.

Readback imports the Exp30 MiniMax 10-step heldout summary:

- Frozen mean mask/boundary/outside PSNR deltas:
  `-0.001068`, `-0.002821`, `-0.006340`.
- EMA mean mask/boundary/outside PSNR deltas:
  `-0.001851`, `-0.003092`, `-0.006033`.
- Step10 visual better count: `0/32`.

Next metric milestone: no-change forensic audit.

## 2026-06-27 No-Change Forensic Audit

- Root-cause status: `MINIMAX_NOCHANGE_CAUSE_UTILITY_SCALE_TOO_WEAK`.
- Frozen parameter delta / param norm ratio: `5.6404525516172905e-06`.
- EMA parameter delta / param norm ratio: `5.630459939756668e-06`.
- Step0/Step10 byte-identical rows: 0/32.
- Mean full abs pixel diff: `0.13143352206508793`.
- Mean mask abs pixel diff: `0.18672874342540607`.
- Mean affected abs pixel diff: `0.1731182035360047`.
- Mean outside abs pixel diff: `0.10850902535158265`.
- Frozen linear utility mean: `0.4999982982873917`.
- EMA linear utility mean: `0.5000003516674042`.

The audit shows nonzero but sub-perceptual checkpoint/output movement.

## 2026-06-27 Inference Sensitivity Positive-Control

- Status: `MINIMAX_INFERENCE_SENSITIVITY_PASS`.
- Rows: 4, using 2 heldout and 2 train rows.
- Identity control max full MAE: `0.0`.
- Perturbed mean full MAE: `0.08821829589193357`.
- Perturbed mean mask MAE: `0.15630244233590715`.
- Perturbed tensors: `16`.
- Perturb scale: `1.01`.
- Adapter scale sweep: `NOT_APPLICABLE_NO_LORA_SCOPE_IN_EXP30`.

The metric result confirms that the MiniMax inference path responds to
transformer-weight changes. The response remains visually subtle, so this
milestone is a path/sensitivity pass only, not a quality-positive adapter gate.
