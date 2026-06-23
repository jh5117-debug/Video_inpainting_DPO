# Exp26 Probe4 Official 49F Inference Result

Date: 2026-06-23 UTC

Runtime root:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623`

## Task

The overnight controller ran official VideoPainter 49-frame inference on the
locked Probe4 VOR-BG formal sources.

Command log:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623/exp26_probe4_official_inference.log`

Output:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623/exp26_probe4_official_inference`

## Configuration

- Base model:
  `/mnt/nas/hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter/ckpt/CogVideoX-5b-I2V`
- Branch checkpoint:
  `/mnt/nas/hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch`
- Manifest:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/49f_probe_7f9ec40/vp2_probe4_49f_masks.jsonl`
- Rows: `4`
- Frames per row: `49`
- Resolution: `720x480`
- Inference steps: `20`
- dtype: `bf16`
- GPU: `2`

## Result

Status: `PASSED`

All four Probe4 rows produced official VideoPainter outputs with exactly
`49` frames:

- `vp2_vor_bg_49f_REAL_ENV181_00006_007_01`
- `vp2_vor_bg_49f_REAL_ENV266_00005_001_01`
- `vp2_vor_bg_49f_REAL_ENV177_00004_003_02`
- `vp2_vor_bg_49f_REAL_ENV222_00105_004_01`

Summary JSON:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623/exp26_probe4_official_inference/probe4_official_inference_summary.json`

This completes the official 49F Probe4 inference gate. Gate16/Gate64 and DPO
training remain unstarted.

