# Exp25 DiffuEraser OR Stack Audit

Date: 2026-06-23

## Decision Summary

The current DiffuEraser smoke failure is **not** a DiffuEraser DPO/main-model LoRA training issue. The SFT/DPO model identity is a full DiffuEraser checkpoint with `brushnet/` and `unet_main/` modules. The failing path is the object-removal inference stack loading PCM inference acceleration weights through diffusers' LoRA loader.

Current status:

- `DE_OFFICIAL_PCM2`: blocked pending a pinned official DiffuEraser environment smoke.
- `DE_CANONICAL_NO_PCM`: historical BR/DAVIS no-PCM path is verified for BR evaluation semantics, but OR no-PCM still requires an explicit Exp25 OR smoke identity before it can become a primary VOR generator.
- Gate128 generation remains blocked. No silent PCM bypass or fallback is allowed.

## Current Project Evidence

Read files:

- `inference/run_OR.py`
- `inference/run_BR.py`
- `diffueraser/diffueraser.py`
- `diffueraser/diffueraser_OR.py`
- `exp15_or_benchmark_davis50/code/infer_diffueraser_or_exp15.py`
- `tools/run_davis50_framewise_protocol_eval.py`

Observed behavior:

- `inference/run_OR.py` sets `ckpt = "2-Step"` and passes `pcm_weights_path` into `DiffuEraser`.
- `diffueraser/diffueraser_OR.py` maps `2-Step` to `pcm_sd15_smallcfg_2step_converted.safetensors`, `num_inference_steps = 2`, and `guidance_scale = 0.0`.
- `diffueraser/diffueraser_OR.py` then calls `self.pipeline.load_lora_weights(pcm_weights_path, weight_name=..., subfolder=mode)`.
- The smoke failure happens inside this PCM load path with `AttributeError: 'UNetMotionModel' object has no attribute 'load_lora_adapter'`.
- `tools/run_davis50_framewise_protocol_eval.py` explicitly reports the target BR evaluation protocol as `raw6, no PCM`.
- `inference/run_BR.py` exposes `--use_pcm`, defaults it to false, and labels target-domain DAVIS evaluation as no PCM.

## Official DiffuEraser Evidence

Official repo clone:

`/home/hj/video_inpainting_third_party/DiffuEraser`

Official commit:

`8e6f279ac7531e27ad1849c6f8dab5372a8597e7`

Official requirements:

- `torch==2.3.1`
- `diffusers==0.29.2`
- `accelerate==0.25.0`
- `transformers==4.41.1`
- `peft==0.13.2`

Official `run_diffueraser.py` also sets `ckpt = "2-Step"` and the official `diffueraser/diffueraser.py` loads PCM weights from `weights/PCM_Weights` via `pipeline.load_lora_weights(...)`.

Current HAL Python environment is not official-pinned:

- `torch 2.9.1`
- `diffusers 0.35.2`
- `peft 0.18.1`
- `transformers 4.57.3`
- `accelerate 1.12.0`

The current PAI smoke environment was previously observed with similarly newer packages (`torch 2.6.0+cu126`, `diffusers 0.37.1`, `peft 0.19.1`, `transformers 5.5.4`, `accelerate 1.13.0`). This version gap is a plausible source of the PCM LoRA loader API mismatch.

## Checkpoint Identity

SFT-48000 core checkpoint:

`/home/hj/Video_inpainting_DPO/weights/diffuEraser/converted_weights_step48000`

Core files:

- `brushnet/config.json`
- `brushnet/diffusion_pytorch_model.safetensors`
- `unet_main/config.json`
- `unet_main/diffusion_pytorch_model.safetensors`

SHA256:

- brushnet config: `99222422dbfe4be5944af70d5d47cc3c28163c5715c2a1a423a7f0d42f104623`
- brushnet weights: `3bdaaed522c3572cfab7d94678f971639e291145048e6dc44310dc184a4c2779`
- unet_main config: `f0e270fa010cc29e2d689624870000bde8dae8b6f988d9f9d141a35d94c2a31e`
- unet_main weights: `c191c4adcc7ff1517dbcabad41a6ad07e340215d0069eaf23a1e687e3f2a24f7`

PCM 2-step sd15 weight:

`/home/hj/Video_inpainting_DPO/weights/PCM_Weights/sd15/pcm_sd15_smallcfg_2step_converted.safetensors`

SHA256:

`4c5f27a727d12146de4b1d987cee3343bca89b085d12b03c45297af05ce88ef4`

## Stack Definitions

### DE_OFFICIAL_PCM2

Definition:

- Official DiffuEraser codepath.
- Official/pinned DiffuEraser dependency versions.
- `ckpt = "2-Step"`.
- PCM file: `pcm_sd15_smallcfg_2step_converted.safetensors`.
- Scheduler follows the official PCM/TCD setup in DiffuEraser.
- ProPainter prior is part of the OR pipeline.

Current status:

`BLOCKED_PENDING_OFFICIAL_PINNED_ENV_SMOKE`

Reason:

The active project environment fails during PCM LoRA load. The next valid action is to run the official example in an isolated official-pinned environment, then adapt an Exp25 wrapper only if the official stack succeeds.

### DE_CANONICAL_NO_PCM

Definition:

- No PCM weights are read.
- Generator identity must include `no_pcm`.
- Steps/scheduler must match the verified no-PCM protocol.
- For BR/DAVIS, the verified protocol is `raw6, no PCM`, hard comp only for metrics.
- For OR/VOR, this cannot be promoted without an OR-specific no-PCM smoke.

Current status:

`BR_VERIFIED_OR_PENDING`

Reason:

The BR no-PCM path is historically verified, but Exp25 needs OR on-policy losers. A BR path must not be used to silently relabel OR outputs.

## Required Next Gate

Before any Gate128 generation:

1. Run `DE_OFFICIAL_PCM2` in an official-pinned environment or mark upstream/assets blocked.
2. Run an explicit `DE_CANONICAL_NO_PCM` OR smoke if no-PCM is considered for policy-matched diagnostics.
3. Choose exactly one primary DiffuEraser OR stack and record `generator_id`.
4. Re-run the fixed six-sample smoke and visual review.

No fallback output is currently eligible for training.
