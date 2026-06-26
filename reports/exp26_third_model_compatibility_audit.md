# Exp26 Third-Model Compatibility Audit

Status: `EXP26_THIRD_MODEL_COMPATIBILITY_AUDIT_COMPLETE`

Scope: static compatibility audit only. No third-model inference, DPO smoke, or training was launched. The audit read local Exp24 backend scaffolding, model-source reports, local third-party code, and current asset state. It did not adopt left CLI commits or touch Exp25/27/28 worktrees.

## Bottom Line

No audited third model is currently `TRUE_DPO_ADAPTER_READY`. The next third-backbone work should be a narrow backend smoke, not training.

| Model | Classification | Adapter verdict | Immediate next action |
| --- | --- | --- | --- |
| ProPainter | `INFERENCE_BASELINE_ONLY;LOSER_GENERATOR_ONLY;NOT_APPLICABLE_NON_DIFFUSION` | no | use as baseline/loser generator only; do not count as third adapter backbone |
| MiniMax-Remover | `ADAPTER_POSSIBLE_NEEDS_TRAINING_FORWARD;BLOCKED_NO_WEIGHTS` | future possible only after native flow-matching loss audit | first run isolated inference baseline smoke; then one-batch native flow-target policy/reference parity, not DPO training |
| EffectErase | `LOSER_GENERATOR_ONLY;INFERENCE_BASELINE_ONLY;ADAPTER_POSSIBLE_NEEDS_TRAINING_FORWARD` | possible future backend, not primary VOR on-policy evidence | use only as diagnostic/baseline; do not use as primary on-policy loser or third adapter until non-VOR validation protocol is defined |
| ROSE | `BLOCKED_NO_TRAINING_CODE;BLOCKED_NO_WEIGHTS` | no current evidence | asset/repo audit only; do not schedule DPO |
| FloED | `ADAPTER_POSSIBLE_NEEDS_TRAINING_FORWARD;BLOCKED_NO_WEIGHTS;BLOCKED_NO_TRAINING_CODE` | future promising if code/weights released | verify release completeness before any backend work |
| VACE | `ADAPTER_POSSIBLE_NEEDS_TRAINING_FORWARD;BLOCKED_NO_WEIGHTS` | possible future, not ready | asset deploy + one-sample inference + native target parity before any DPO |
| CoCoCo | `ADAPTER_POSSIBLE_NEEDS_TRAINING_FORWARD;BLOCKED_NO_WEIGHTS;BLOCKED_NO_TRAINING_CODE` | possible but training code/weights blocker | best next third-adapter candidate after weights: one-batch native loss/parity smoke; no long training |

## Key Findings

- ProPainter is a useful baseline / flow-prior / loser generator, but it is non-diffusion and should not be counted as a Diffusion-DPO adapter backbone.
- MiniMax-Remover has a public inference pipeline and transformer forward, but current local cache lacks usable weights and no full training/minimax objective is locked. It should be an inference baseline first.
- EffectErase is a strong OR diagnostic/baseline candidate, but it is VOR-trained and CC BY-NC; it must not be used as a primary on-policy VOR loser. Its Wan/DiffSynth training utilities make future adapter work plausible, not ready.
- ROSE is not locally available; treat as future benchmark/affected-region metric work only.
- FloED is scientifically interesting for a flow-guided diffusion adapter, but local code/weights are not confirmed.
- VACE is a possible future Wan/flow-matching adapter, but currently blocked by assets and native target parity.
- CoCoCo is the lowest-risk future true-adapter candidate if weights/dependencies are solved, because it is SD/UNet-like and has local forward code. However, official training code is not released/locked, so it is not ready for DPO training.

## Recommendation

Do not immediately train a third model. The next minimum useful experiment is a CoCoCo or MiniMax isolated inference smoke plus native policy/reference one-batch target parity. If the goal is a third true adapter backbone, prioritize CoCoCo after weights/dependency resolution; if the goal is an OR baseline/loser generator, prioritize MiniMax and ProPainter.

The current paper claim should remain: DiffuEraser plus VideoPainter provide cross-backbone adapter evidence, not universal adapter evidence.
