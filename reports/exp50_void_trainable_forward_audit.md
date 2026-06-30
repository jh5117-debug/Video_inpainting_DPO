# Exp50 VOID Trainable Forward Audit

Status: `VOID_TRAINING_FORWARD_HEAVY_BUT_POSSIBLE`.

This is a read-only audit of the official VOID repo. No official source files were edited, no training was launched, and no model weights were loaded for full inference.

## Key Findings

- Policy model: `CogVideoXTransformer3DModel` from the CogVideoX-Fun transformer subfolder.
- Trainable surface: transformer modules selected by `--trainable_modules`; official VOID shell scripts use `--trainable_modules "."`, so the default is effectively heavy transformer fine-tuning.
- Frozen modules: VAE and T5 text encoder are frozen with `requires_grad_(False)`.
- Target: scheduler epsilon or v-prediction velocity target, depending on scheduler `prediction_type`.
- Loss: mean MSE between predicted noise and target; optional temporal/motion sub-loss exists.
- Quadmask: [0, 63, 127, 255], with 63 representing object/affected overlap.
- Pass1/pass2: pass2 warped-noise training depends on a pass1 checkpoint and warped noise generated from pass1 outputs.
- Save/reload: supported by Accelerate save/load hooks and `resume_from_checkpoint`.

## Required Answers

1. Policy model is `CogVideoXTransformer3DModel`.
2. Trainable modules are selected transformer parameters; official scripts select all transformer names via `.`.
3. Frozen modules are VAE and text encoder; unselected transformer parameters remain frozen.
4. Target parameterization is epsilon or v-prediction velocity from the scheduler.
5. Loss is MSE noise/velocity prediction loss, optionally mixed with motion sub-loss.
6. Quadmask is encoded as 0 pure remove, 63 overlap, 127 affected/modify, 255 keep.
7. Pass1 produces the first VOID model/output; pass2 uses pass1 checkpoint/output-derived warped noise.
8. Freeze reference is feasible in isolated code, but official DPO reference-policy plumbing is not present.
9. One-step optimizer is feasible through `--max_train_steps=1`.
10. Save/reload checkpoint is supported.
11. LoVI-DPO adapter is not out-of-box; LoRA utilities exist, but DPO would need isolated implementation.
12. VOID is not merely inference baseline; official trainable forward exists.

## Decision

`VOID_TRAINING_FORWARD_HEAVY_BUT_POSSIBLE`: continue with data adapter/Gate8, but do not claim VOID positive and do not run adapter micro gates unless downstream smoke gates pass.
