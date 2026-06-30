# Exp50 VOID SFT Forward Parity

Status: `VOID_SFT_FORWARD_PARITY_EXPLAINED`

## Run

- Sample: `BLENDER_CON001_00636`
- Device: `cuda:0`
- dtype: `torch.bfloat16`
- Requested frames: 17
- Effective frames after official patch-size truncation: 13
- Resolution: 672x384
- Timestep: 500
- Seed: 1234
- Prediction type: `v_prediction`
- SFT loss: 0.035282157361507416
- Loss finite: True

## Equivalence

The wrapper mirrors the official `scripts/cogvideox_fun/train.py` path for the core forward: VAE encode target video, construct VAE-mask inpaint latents from condition and quadmask, sample scheduler noise/timestep, build the scheduler target (`epsilon` or `v_prediction`), run `CogVideoXTransformer3DModel`, and compute mean MSE. The official helper is not directly callable without entering the full Accelerator training loop, so this is marked `EXPLAINED` rather than strict byte-for-byte parity.

## Safety

No training, optimizer step, DPO, VOR-Eval, hard comp, official source modification, or deepspeed install was performed.
