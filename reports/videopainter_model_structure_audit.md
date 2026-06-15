# VideoPainter Model Structure Audit

Date: 2026-06-15

## Verdict

VideoPainter is diffusion / denoising based and can support a **direct
Diff-DPO design in principle**, but a runnable adapter trainer is **blocked**
until an isolated pair-dataset + policy/reference training loop is implemented.

Do not run upstream VideoPainter training as the adapter gate. Upstream training
does not compute `m_w`, `m_l`, `m_w_ref`, `m_l_ref`, normalized gaps, or DPO
diagnostics.

## Audited Repo

Local repo:

```text
/home/hj/dpo-2-1-exp/third_party_baselines/VideoPainter
```

Remote:

```text
https://github.com/TencentARC/VideoPainter.git
```

Commit:

```text
bbab6cd5cd5cb89f0e2444305c32fd74a010ae0a
```

Key training entry:

```text
train/VideoPainter.sh
train/train_cogvideox_inpainting_i2v_video.py
```

## What VideoPainter Trains

The upstream training loop loads:

- `CogVideoXTransformer3DModel`
- `AutoencoderKLCogVideoX`
- `CogvideoXBranchModel`
- `CogVideoXDPMScheduler`
- T5 tokenizer / text encoder

It freezes the base components and trains the extra branch:

```text
text_encoder.requires_grad_(False)
transformer.requires_grad_(False)
vae.requires_grad_(False)
branch.requires_grad_(True)
```

## Inputs

The official dataset path expects a CSV and raw videos:

```text
meta_file_path: pexels_videovo_train_dataset.csv
instance_data_root: raw video root
```

Each row provides:

```text
video_path, start_frame, end_frame, fps, mask_id, prompt
```

The mask is loaded from an `all_masks.npz` tree inferred from the CSV location.

Training batches contain:

```text
pixel_values
conditioning_pixel_values
masks
input_ids / prompts
```

## Outputs And Loss

The training loop samples noise and timesteps:

```text
noise = torch.randn_like(model_input)
timesteps = torch.randint(...)
noisy_video_latents = scheduler.add_noise(model_input, noise, timesteps)
```

The branch and transformer forward produce `model_output`; then:

```text
model_pred = scheduler.get_velocity(model_output, noisy_video_latents, timesteps)
target = model_input
```

The official loss is a weighted denoising MSE plus mask inpainting loss:

```text
loss = mean(weights * (model_pred - target)^2)
inpainting_loss = mean(weights * (model_pred * masks - target * masks)^2)
loss = loss + inpainting_loss_weight * inpainting_loss
```

This is a diffusion-style latent denoising objective. Therefore a direct
Diff-DPO loss is mathematically definable.

## Direct Diff-DPO Feasibility

Required quantities:

```text
m_w     = policy loss on GT winner
m_l     = policy loss on generated loser
m_w_ref = frozen reference loss on GT winner
m_l_ref = frozen reference loss on generated loser
```

These are feasible only if an isolated adapter trainer:

1. reads a pair manifest with winner / loser / mask;
2. encodes winner and loser through the same VAE path;
3. samples one shared timestep/noise schedule per pair;
4. runs the trainable policy branch/transformer on winner and loser;
5. runs a frozen reference branch/transformer on winner and loser under
   `torch.no_grad()`;
6. computes region-weighted MSE at latent resolution;
7. computes the Exp11 outer b0.75 normalized-gap DPO objective;
8. writes `dpo_diagnostics.csv`.

## Current Pair Data Mismatch

The current project DPO manifest on PAI has frame-directory paths:

```text
win_video_path: frame directory with 00000.png ...
final_loser_video_path: frame directory with 00000.png ...
mask_path: frame directory with 00000.png ...
```

Example fields:

```text
win_video_path
final_loser_video_path
raw_loser_video_path
mask_path
mask_convention = png_255_inpaint_region_0_keep_region
num_frames = 16
height = 320
width = 512
```

This does not match upstream VideoPainter's CSV + raw-video + `all_masks.npz`
loader. A new pair dataloader is required under:

```text
exp14_adapter_videopainter/code/
```

## Reference Model Feasibility

A frozen reference is feasible in principle by loading the same branch /
transformer checkpoint twice:

```text
policy: trainable branch, frozen base transformer
reference: frozen branch + frozen base transformer
```

But it is not implemented upstream and memory has not been measured on PAI.

## Checkpoint And Eval

Upstream training can save branch checkpoints through Accelerate hooks.
Upstream inference and eval scripts exist, but they use VideoPainter's own data
layout. A thin eval adapter is still required to produce the project-standard
four-column visualization and metrics.

## Final Decision

```text
adapter_type = direct_diff_dpo_blocked_pending_isolated_trainer
```

Do not launch gate2000 until the isolated trainer exists. Running the upstream
official VideoPainter training command would not be a DPO adapter experiment.

