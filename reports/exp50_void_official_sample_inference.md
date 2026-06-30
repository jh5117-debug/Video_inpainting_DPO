# Exp50 VOID Official Sample Inference

Timestamp: 2026-06-30T17:02:21+08:00

Status: `VOID_OFFICIAL_SAMPLE_INFERENCE_PASS`

## Command

- Official script: `inference/cogvideox_fun/predict_v2v.py`
- Sample: `lime` from official repo `sample/`
- Pass: Pass1 only
- Base model: `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/void/CogVideoX-Fun-V1.5-5b-InP`
- Checkpoint: `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/void/netflix_void-model/void_pass1.safetensors`
- Output root: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp50_pai_void_adapter_feasibility/f1_official_sample_lime`
- Log: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp50_pai_void_adapter_feasibility/f1_official_sample_lime.log`

## Runtime Notes

- Return code: `0`
- Raw output: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp50_pai_void_adapter_feasibility/f1_official_sample_lime/lime-fg=-1-0001.mp4`
- Tuple output: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp50_pai_void_adapter_feasibility/f1_official_sample_lime/lime-fg=-1-0001_tuple.mp4`
- Raw frame count: 85
- Bundled ffmpeg shim: `ffmpeg_symlink=/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp50_pai_void_adapter_feasibility/ffmpeg_bin/ffmpeg`
- Checkpoint load clean: True
- Sampling reached 30/30: True

## Visual Review

Codex opened `quick_visual_sheet.png` and `quick_tuple_sheet.png`. The output is nonblank, temporally sampled frames are present, and the tuple view is aligned. This is a technical official-sample smoke only, not a VOR quality claim.

## Safety

No training, no optimizer step, no VOR-Eval, no hard comp, no VOID positive claim.
