# VideoPainter Weight Resolution Report

Date: Tue Jun 16 04:55:36 CST 2026
Host: dsw-753014-dc85766cb-4v2jj
Root: /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate
VideoPainter repo: /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter

## Disk
Filesystem                                                                           Size  Used Avail Use% Mounted on
rund:TS9haq7t:cpfs-01000vwrt8a6usy68r6wu-000001.cn-shanghai.cpfs.aliyuncs.com:/pku/   70T   69T  1.6T  98% /mnt/workspace
172.28.48.25:/                                                                        10P  6.6T   10P   1% /mnt/nas
172.28.48.25:/                                                                        10P  6.6T   10P   1% /mnt/nas

## Tooling
/usr/bin/git
/usr/local/bin/huggingface-cli
huggingface_hub 1.11.0
HF_TOKEN_set=yes
HUGGINGFACE_HUB_TOKEN_set=yes

## Existing path checks
MISSING /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter/ckpt/CogVideoX-5b-I2V
MISSING /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch
MISSING /mnt/workspace/hj/nas_hj/weights/CogVideoX-5b-I2V
MISSING /mnt/workspace/hj/nas_hj/weights/VideoPainter
MISSING /mnt/workspace/hj/nas_hj/weights/videopainter
MISSING /mnt/nas/hj/weights/CogVideoX-5b-I2V
MISSING /mnt/nas/hj/weights/VideoPainter
MISSING /mnt/nas/hj/weights/videopainter
MISSING /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/third_party/VideoPainter/ckpt/CogVideoX-5b-I2V
MISSING /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch
MISSING /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp09_10_11_pai_sync/third_party/VideoPainter/ckpt/CogVideoX-5b-I2V
MISSING /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp09_10_11_pai_sync/third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch
MISSING /mnt/nas/hj/H20_Video_inpainting_DPO/third_party/VideoPainter/ckpt/CogVideoX-5b-I2V
MISSING /mnt/nas/hj/H20_Video_inpainting_DPO/third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch
MISSING /mnt/nas/hj/H20_Video_inpainting_DPO_exp09_10_11_pai_sync/third_party/VideoPainter/ckpt/CogVideoX-5b-I2V
MISSING /mnt/nas/hj/H20_Video_inpainting_DPO_exp09_10_11_pai_sync/third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch
MISSING /mnt/nas/hj/official_repos/VideoPainter/ckpt/CogVideoX-5b-I2V
MISSING /mnt/nas/hj/official_repos/VideoPainter/ckpt/VideoPainter/checkpoints/branch
MISSING /mnt/nas/hj/external/VideoPainter/ckpt/CogVideoX-5b-I2V
MISSING /mnt/nas/hj/external/VideoPainter/ckpt/VideoPainter/checkpoints/branch
MISSING /home/hj/dpo-2-1-exp/third_party_baselines/VideoPainter/ckpt/CogVideoX-5b-I2V
MISSING /home/hj/dpo-2-1-exp/third_party_baselines/VideoPainter/ckpt/VideoPainter/checkpoints/branch
MISSING /home/hj/weights/CogVideoX-5b-I2V
MISSING /home/hj/weights/VideoPainter
MISSING /home/hj/Video_inpainting_DPO/third_party/VideoPainter/ckpt/CogVideoX-5b-I2V
MISSING /home/hj/Video_inpainting_DPO/third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch
MISSING /home/nvme01/H20_Video_inpainting_DPO/third_party/VideoPainter/ckpt/CogVideoX-5b-I2V
MISSING /home/nvme01/H20_Video_inpainting_DPO/third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch
MISSING /home/nvme01/H20_Video_inpainting_DPO/weights/CogVideoX-5b-I2V
MISSING /home/nvme01/H20_Video_inpainting_DPO/weights/VideoPainter

## Limited exact-name search
### /mnt/workspace/hj/nas_hj
### /mnt/nas/hj

## HF download attempt
Date: Tue Jun 16 04:57:24 CST 2026
ckpt_dir: /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter/ckpt
videopainter_download_rc=1
VIDEO_PAINTER_DOWNLOAD_FAILED

## HF download attempt using hf CLI
Date: Tue Jun 16 04:57:47 CST 2026
/usr/local/bin/hf
1.11.0
videopainter_hf_download_rc=1
VIDEO_PAINTER_HF_DOWNLOAD_FAILED

## Final Weight Resolution Status

status: blocked_missing_weights_and_pai_hf_network_unreachable

Existing-weight search result:

- CogVideoX-5b-I2V was not found on PAI/NAS/HAL-mounted/H20-like paths.
- VideoPainter branch checkpoint was not found on PAI/NAS/HAL-mounted/H20-like paths.

HF download result:

- `huggingface-cli` is deprecated in this PAI image and exits with failure.
- `hf download TencentARC/VideoPainter` failed before file download with:
  `httpx.ConnectError: [Errno 101] Network is unreachable`.
- A separate `hf download THUDM/CogVideoX-5b-I2V` attempt was also started, but
  it produced no files/log output after repeated polling and was interrupted by
  the agent as another PAI network-blocked attempt.

Decision:

- Do not run trainer preflight.
- Do not launch gate2000.
- Do not run upstream VideoPainter official training as a fallback.

Required user/admin action:

Place the following directories on PAI, preferably under:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter/ckpt/
```

Required final layout:

```text
ckpt/CogVideoX-5b-I2V/model_index.json
ckpt/CogVideoX-5b-I2V/transformer/
ckpt/CogVideoX-5b-I2V/vae/
ckpt/CogVideoX-5b-I2V/tokenizer/
ckpt/CogVideoX-5b-I2V/text_encoder/
ckpt/VideoPainter/checkpoints/branch/config.json
ckpt/VideoPainter/checkpoints/branch/diffusion_pytorch_model.safetensors
```

Suggested rsync from a machine that can access Hugging Face:

```bash
rsync -az /path/to/CogVideoX-5b-I2V/ \
  root@47.103.26.60:/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter/ckpt/CogVideoX-5b-I2V/

rsync -az /path/to/VideoPainter/ \
  root@47.103.26.60:/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter/ckpt/VideoPainter/
```

After weights are present, rerun:

```bash
cd /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate
bash exp14_adapter_videopainter/scripts/launch_videopainter_adapter_gate2000_pai.sh
```

## Separate CogVideoX HF download attempt
Date: Tue Jun 16 05:03:38 CST 2026

## CogVideoX download final note
The separate CogVideoX hf-download attempt produced no files and no log output
after repeated polling, consistent with the same PAI network blockage. The agent
interrupted only this self-started download process; no user/training process
was killed.
