# Exp59 Official VOID Inference Protocol Audit

Date: 2026-07-02

Status: `EXP59_OFFICIAL_INFERENCE_PROTOCOL_READY`

## Official Files Audited

- `/mnt/nas/hj/H20_Video_inpainting_DPO/third_party/VOID/Netflix_void-model/README.md`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/third_party/VOID/Netflix_void-model/inference/cogvideox_fun/predict_v2v.py`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/third_party/VOID/Netflix_void-model/inference/cogvideox_fun/inference_with_pass1_warped_noise.py`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/third_party/VOID/Netflix_void-model/config/quadmask_cogvideox.py`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/third_party/VOID/Netflix_void-model/videox_fun/utils/utils.py`

## Protocol

Official pass1 command pattern:

```bash
python inference/cogvideox_fun/predict_v2v.py \
  --config config/quadmask_cogvideox.py \
  --config.data.data_rootdir=<input_root> \
  --config.experiment.run_seqs=<comma-separated sequences> \
  --config.experiment.save_path=<output_root> \
  --config.video_model.model_name=/mnt/nas/hj/H20_Video_inpainting_DPO/weights/void/CogVideoX-Fun-V1.5-5b-InP \
  --config.video_model.transformer_path=/mnt/nas/hj/H20_Video_inpainting_DPO/weights/void/netflix_void-model/void_pass1.safetensors
```

Input folder format:

```text
sample_dir/
  input_video.mp4
  quadmask_0.mp4
  prompt.json
```

`prompt.json` must contain a single `bg` key describing the scene after the object is removed. It must not describe the removal operation.

## Resolution And Frame Count

The Exp58B Kubric Gate8 videos are native 128x128, 24 frames, 8 fps. Official `quadmask_cogvideox.py` defaults are:

- `config.data.sample_size = "384x672"`
- `config.data.max_video_length = 197`
- `config.data.fps = 12`
- `config.video_model.temporal_window_size = 85`
- `config.video_model.sampler_name = "DDIM_Origin"`
- `config.system.weight_dtype = torch.bfloat16`
- `config.system.seed = 42`
- `config.system.gpu_memory_mode = "model_cpu_offload_and_qfloat8"`

The official `get_video_mask_input` path reads `input_video.mp4`, reads `prompt.json["bg"]`, then resizes the input video and mask with `F.interpolate(..., sample_size)`. Therefore 128x128 inputs are accepted through official preprocessing and will be upsampled to the configured output resolution.

Short 24-frame clips are padded by the official utility to the temporal window. This is official preprocessing, but it is a diagnostic caveat: output videos will be at the official processing length/resolution, while GT `rgb_removed` is native 128x128/24F.

## Mask Handling

The README defines quadmask values:

- `0`: primary object
- `63`: primary + affected overlap
- `127`: affected region
- `255`: background keep

The official loader quantizes the mask to these four values and inverts it internally before model use. Exp59 materialization should preserve the source quadmask values exactly before official preprocessing.

## Inference Parameters

- Pass: pass1 only
- Base model: `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/void/CogVideoX-Fun-V1.5-5b-InP`
- Pass1 checkpoint: `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/void/netflix_void-model/void_pass1.safetensors`
- Scheduler: `DDIM_Origin`
- Script call denoising steps: `30` in `predict_v2v.py`
- Guidance scale: `1.0`
- Seed: `42`
- dtype: `bfloat16`
- Memory mode: `model_cpu_offload_and_qfloat8`

PAI GPU audit at protocol time showed GPU0-7 idle, each `NVIDIA L20X` with 143771 MiB total VRAM.

## Metric Caveat For Later Milestones

Official output will be produced at the configured sample size. Metrics against `rgb_removed` must explicitly use a common resolution. Exp59 should report this as a diagnostic protocol choice rather than silently treating native and official resolutions as identical.

## Safety

- Training run: no
- Preference forward: no
- Zero-gap: no
- One-step: no
- 10-step: no
- Official VOID source modified: no
- `inference/metrics.py` modified: no
- Shared trainer modified: no
