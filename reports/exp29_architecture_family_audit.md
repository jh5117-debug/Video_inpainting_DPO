# Exp29 Architecture Family Audit

Date: 2026-06-26

Status: `EXP29_ARCHITECTURE_FAMILY_AUDIT_COMPLETED`

This audit was completed before any new EffectErase inference smoke, MiniMax
expanded candidate generation, trainable-forward gate, or training task.

## Core Finding

The Exp29 models do not share one universal SD1.5 trainer. The correct framing is
model-specific backend adapters:

- DiffuEraser: SD1.5 / UNet-style latent diffusion with BrushNet / MotionModule
  stages.
- VideoPainter: CogVideoX / VideoPainter video model with velocity-style target.
- MiniMax-Remover: Wan2.1 / DiT / flow-matching remover with velocity target
  `epsilon - z0`.
- EffectErase: Wan / DiT / flow-style removal pipeline with LoRA and remove
  condition adapter; currently OR baseline / diagnostic until inference and any
  training-forward gates are proven.

Therefore current language remains:

`CROSS_BACKBONE_ADAPTER_EVIDENCE_CONFIRMED`

but not:

`UNIVERSAL_ADAPTER`

## Evidence

### DiffuEraser

Prior implementation/report evidence identifies DiffuEraser as an SD1.5-style
latent diffusion inpainting stack:

- UNet2D main model plus BrushNet condition residuals.
- Stage2 MotionModule for temporal adaptation.
- Preference losses compare policy/reference noise-prediction MSE on shared
  noise/timestep.

Role: confirmed LoVI-DPO adapter backbone.

### VideoPainter

Exp26 evidence identifies VideoPainter as a VideoPainter / CogVideoX video
branch, not merely SD1.5:

- `train_videopainter_dpo_adapter.py` loads `CogVideoXTransformer3DModel`,
  `AutoencoderKLCogVideoX`, and `CogVideoXDPMScheduler`.
- Reports describe official VideoPainter training as sampling noise/timestep,
  adding noise to video latents, predicting velocity, and computing denoising
  loss.
- Exp26 completed Step0/1/10/50 and independent shadow-dev confirmation.

Role: confirmed LoVI-DPO adapter backbone.

### MiniMax-Remover

Exp29 MiniMax adapter code explicitly uses a flow target:

```python
zt = t * eps + (1 - t) * z0
target = eps - z0
pred = model(hidden_states=hidden, timestep=(t * 1000.0).float())[0]
loss = mse(pred, target)
```

This is not the DiffuEraser epsilon/noise target. The current Exp29 MiniMax
zero-gap/one-step/10-step gates did not directly reuse DiffuEraser epsilon DPO
without conversion.

Role: third-backbone flow-style adapter candidate. Current quality status is
`MINIMAX_DATA_YIELD_INSUFFICIENT`, not quality-positive.

### EffectErase

EffectErase official code uses Wan remove inference:

- `examples/remove_wan/infer_remove_wan.py`
- `WanRemovePipeline`
- `task="remove"`
- `model_manager.load_lora_v2(...)`
- remove condition adapter loaded strictly from the LoRA state.

Its generic DiffSynth/Wan training utilities expose flow matching:

- `scheduler.add_noise(sample, noise, timestep) = (1 - sigma) * sample + sigma * noise`
- `scheduler.training_target(sample, noise, timestep) = noise - sample`

EffectErase is still not adapter-ready in Exp29 because the recovered weights
only unblock inference smoke. No EffectErase trainable-forward, zero-gap, or
one-step gate has run.

Role: OR strong baseline / diagnostic candidate. Because the model is trained on
VOR, VOR results are not primary scientific adapter evidence.

### ProPainter

ProPainter remains a propagation / optical-flow / transformer-style inpainting
baseline and loser generator. It does not expose a native diffusion noise,
timestep, or flow target for the current LoVI / SDPO adapter objective.

Role: baseline / loser generator, not a true Diffusion-DPO adapter under the
current objective.

### ROSE

ROSE is treated as an OR benchmark / affected-region metric candidate and
possible future adapter only if local training-forward code and weights are
available. No Exp29 true-adapter gate has run for ROSE.

### FloED

FloED remains a future candidate only if complete code, weights, and a real
training-forward target are available. No Exp29 true-adapter gate has run for
FloED.

## Safety Decision

`MINIMAX_GATE_INVALID_TARGET_MISMATCH` is not triggered for the current Exp29
MiniMax code path, because the audited script computes `epsilon - z0` flow
velocity. Future MiniMax recipe and 30-step gates still remain blocked until the
expanded data-yield gate can build scene-disjoint `train16` and `heldout16`.

## Required Language

Use:

- model-specific backend adapter
- cross-backbone adapter evidence across DiffuEraser and VideoPainter
- MiniMax as flow-style third-backbone candidate
- EffectErase as VOR-confounded OR baseline / diagnostic until non-confounded
  evidence exists

Do not use:

- one-click universal adapter
- all models supported
- final SOTA
- top-conference novelty confirmed
