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

## 2026-06-27 Trainable-Scope Audit

- Status: `MINIMAX_TRAINABLE_SCOPE_CURRENT_OK`.
- Checkpoint tensor count: `461`.
- Total parameter count represented: `1127055424`.
- LoRA/adapter tensor count: `0`.
- Attention q/k/v/out tensors: `60` each.
- MLP tensors: `120`.
- MLP parameter count: `826068480`.

No metric-positive adapter result is implied. The audit only rules out a too
small or ignored trainable scope.

## 2026-06-27 Winner-SFT Positive-Control

- Status: `MINIMAX_POSITIVE_CONTROL_PASS_HELDOUT_QUALITY_NEGATIVE`.
- Winner-SFT loss decreased for all three AdamW recipes:
  - LR `1e-5`: `0.7092440128 -> 0.0127931200`.
  - LR `3e-5`: `0.7092440128 -> 0.0181513224`.
  - LR `1e-4`: `0.7092440128 -> 0.1104465276`.
- Step10 parameter-delta probes were nonzero:
  `1.4444718289e-05`, `4.1587465276e-05`, and `0.0002197052693`.
- Heldout mean mask PSNR deltas were negative:
  `-0.2448377791`, `-0.8897026274`, `-4.2619560566`.
- Heldout mean boundary PSNR deltas were negative:
  `-0.6611956197`, `-2.0405589461`, `-6.4308968032`.

The result proves update sensitivity, not quality improvement.

## 2026-06-27 Bad-Noise / Hard-Timestep Miner

- Status: `MINIMAX_BAD_NOISE_STATES_READY`.
- Model update: false.
- Rows scanned: `32` train, `16` heldout.
- Candidate states per row: `16`.
- Total candidate-state CSV rows: `768`.
- Timesteps: `0.15`, `0.35`, `0.55`, `0.75`.
- Train winner-advantage mask mean: `0.053676288894166646`.
- Heldout winner-advantage mask mean: `0.030786066912696697`.
- Train state manifest SHA256:
  `fbadd0d2565c4bb49245931742215c4d074c9834b369342398058b4ed9732047`.
- Heldout state manifest SHA256:
  `947f6c0f660229f1da92cb756ee7e03cda4b2215d1ae8f154999574b590ec1fb`.

These are frozen residual/state-selection diagnostics only. No output-quality
metric, PSNR gate, LPIPS gate, or visual quality pass is implied.
