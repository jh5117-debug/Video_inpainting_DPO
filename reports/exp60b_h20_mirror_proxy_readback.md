# Exp60B H20 Mirror / Proxy Readback

Status: `EXP60B_H20_HF_MIRROR_READY`

## Git Readback

- Branch: `research/exp60b-videopainter-vpdata-d3mask-pai-20260702`
- Start HEAD: `d668a7c07ffd9ce9d61e13307934886c1f3b3db9`
- Worktree status before this report: clean
- Existing plan manifests:
  - `manifests/exp60b_vpdata_train1000_sources_h20.jsonl`
  - `manifests/exp60b_vpdata_test100_sources_h20.jsonl`

## Manifest Verification

- Train rows: 1,000
- Test rows: 100
- Train/test `source_video_id` overlap: 0
- Existing downloaded raw videos before continuation: 0
- Source filter: `pexels_only`
- No masks, loser generation, DPO, inference, training, or GPU work has run.

## H20 Network Audit

H20 was reached through the PAI relay:

`HAL -> hj@47.103.26.60 -> ubuntu@27.190.15.128`

H20 details:

- Hostname: `instance-afs92r3e`
- User: `ubuntu`
- `/home/nvme01`: 3.4T total, 1.2T available, 66% used
- Python: `python3 3.10.6`
- `python`: missing

HTTP tests with `HF_ENDPOINT=https://hf-mirror.com`:

- `https://hf-mirror.com`: HTTP 200
- `https://huggingface.co`: network unreachable
- `https://www.pexels.com`: HTTP 403

Clash files:

- `/home/nvme01/clash-for-linux/clash.sh`: exists
- `/home/nvme01/clash-for-linux/start.sh`: exists

## Interpretation

Hugging Face metadata access should use `hf-mirror.com`, not `huggingface.co`,
on H20. Pexels raw-video URLs are separate from Hugging Face and may still need
direct URL testing and possibly the H20 clash proxy.

No token was printed or written.

