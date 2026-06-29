# Exp49 ROSE Code / Adapter Feasibility Audit

Status: `ROSE_TRAINING_FORWARD_BLOCKED`

Generated: 2026-06-30T07:45:42.816509+08:00
Host: `dsw-753014-85f54df947-bkp7h`
Branch: `research/exp49-pai-rose-adapter-feasibility-20260629`
Commit: `ac6ffec7bd4aeedb31e98c5ad6b20b43b6e8b0b8`
ROSE repo: `/mnt/nas/hj/H20_Video_inpainting_DPO/third_party/ROSE/Kunbyte-AI_ROSE`
ROSE commit: `6be41c5420bf331c6d491277d5a6feaf9b3a779a`

## Decision

ROSE is not promoted to `ROSE_TRUE_ADAPTER_FEASIBLE` in Milestone D. The released official repo contains an inference entrypoint, a differentiable Wan transformer, dataset classes, and LoRA utilities, but it does not release an executable training script, optimizer/backward loop, explicit loss, or explicit training target construction. Therefore this milestone marks adapter feasibility as `ROSE_TRAINING_FORWARD_BLOCKED`.

This does not mean ROSE is inference-only forever. It means a future Exp49 wrapper would first need to reconstruct the ROSE-native FlowMatch target and prove zero-gap / one-step / strict reload before claiming adapter feasibility.

## Architecture Table

| Component | File | Finding | Adapter impact |
| --- | --- | --- | --- |
| `model_family` | `README.md; configs/wan2.1/wan_civitai.yaml` | Wan2.1-Fun-1.3B-InP based video inpainting diffusion transformer; README says training trains WanTransformer3D and freezes other parts. | Backbone is conceptually adapter-suitable, but released code does not include executable training recipe. |
| `inference_entrypoint` | `inference.py:74-155` | Only CLI entrypoint is inference-oriented; loads base, transformer, scheduler, VAE, text/CLIP encoders and runs with torch.no_grad. | Usable for baseline smoke after path wrapper/symlinks; not a training entrypoint. |
| `pipeline` | `rose/pipeline/pipeline_wan_fun_inpaint.py:468-729` | WanFunInpaintPipeline.__call__ is decorated with @torch.no_grad and runs denoising loop through scheduler.step. | Official pipeline must not be used directly for gradient training without an isolated wrapper. |
| `target_parameterization` | `rose/pipeline/pipeline_wan_fun_inpaint.py:557-699; configs/wan2.1/wan_civitai.yaml` | FlowMatchEulerDiscreteScheduler with num_train_timesteps=1000, shift=5.0; transformer prediction is consumed by scheduler.step as noise_pred. | Training target is not explicitly released; likely flow/velocity style residual must be reconstructed before any LoVI-DPO gate. |
| `trainable_forward` | `rose/models/wan_transformer3d.py:810-1026` | WanTransformer3DModel.forward is ordinary PyTorch forward and supports gradient checkpointing when grad is enabled. | A custom Exp49 wrapper can test gradients later, but Milestone D does not prove optimizer training. |
| `trainable_modules` | `README.md:226-240; rose/utils/lora_utils.py:158-366` | README says only WanTransformer3D is trained; LoRANetwork targets WanTransformer3DModel and can prepare optimizer params. | Trainable scope is plausible for transformer LoRA/full transformer, but no official training script applies it. |
| `frozen_modules` | `inference.py:99-134` | Tokenizer, T5 text encoder, CLIP image encoder, VAE, scheduler, transformer are assembled; README says non-transformer parts are frozen during training. | Reference/frozen split is feasible in principle but needs Exp49 wrapper proof. |
| `checkpoint_load` | `rose/models/wan_transformer3d.py:1080-1192` | WanTransformer3DModel.from_pretrained loads config and safetensors/bin weights. | Strict reload for base transformer appears feasible. |
| `adapter_save_reload` | `rose/utils/lora_utils.py:275-339` | LoRANetwork has load_weights and save_weights for safetensors/torch state_dict. | LoRA save/reload is feasible if a custom training wrapper reaches one-step gate. |
| `dataset_loader` | `rose/data/dataset_image_video.py:186-362; 365-589` | ImageVideoDataset and ImageVideoControlDataset read csv/json annotations and synthesize random inpaint masks. | Native dataset does not directly consume VOR-OR manifests; an Exp49 adapter dataset would be required. |
| `mask_polarity` | `rose/pipeline/pipeline_wan_fun_inpaint.py:592-642; rose/data/dataset_image_video.py:347-351` | Dataset uses mask=1 as masked region; inference pipeline binarizes mask_video and then builds latent mask from 1 - mask_condition. | VOR mask polarity must be tested in inference smoke before any training gate. |
| `difference_mask_predictor` | `rose/models/diff_mask_predictor.py:6-42` | DiffMaskPredictor class exists, but grep found no integration call outside its own definition. | Side-effect predictor is not an exposed training/inference hook in released code. |
| `official_training_release` | `repo tree + grep` | No train/finetune script or optimizer/backward loop found. Candidates: none | Blocks ROSE_TRUE_ADAPTER_FEASIBLE status in Milestone D. |

## Key Evidence

- `inference.py` hard-codes local base/transformer roots (`models/Wan2.1-Fun-1.3B-InP`, `weights/transformer`) and runs validation samples inside `with torch.no_grad()`.
- `WanFunInpaintPipeline.__call__` is decorated with `@torch.no_grad()`, so the official pipeline cannot be directly reused as a training forward.
- `WanTransformer3DModel.forward()` is a normal PyTorch forward and can run with gradients when called outside the no-grad pipeline.
- `LoRANetwork` supports `WanTransformer3DModel`, `prepare_optimizer_params`, `save_weights`, and `load_weights`.
- `DiffMaskPredictor` is defined but not integrated in the released inference/training path.
- No official train/finetune script, `accelerator.backward`, or `optimizer.step` loop was found in the released repo.

## Gate Consequence

- Milestone E official inference remains blocked by `ROSE_ENV_PARTIAL` until the user accepts Python 3.10 as a practical env or a true Python 3.12 Torch env is created.
- Milestone G trainable forward / zero-gap / one-step must not run yet because `ROSE_TRUE_ADAPTER_FEASIBLE` was not reached.

## Safety

No inference, training, optimizer step, checkpoint update, VOR-Eval use, H20 action, or official ROSE source modification was performed.
