# VideoDPO H20 VideoInpainting Adapter

This folder stores a patch for the separate `/home/nvme01/VideoDPO` checkout.
It lets open-source VideoDPO train on the H20 VideoInpainting DPO data root:

`/home/nvme01/H20_Video_inpainting_DPO/DPO_Finetune_Data_Multimodel_v1`

Apply on H20 after pulling this repo:

```bash
cd /home/nvme01/H20_Video_inpainting_DPO
git pull

cd /home/nvme01/VideoDPO
git status --short
mkdir -p /home/nvme01/H20_Video_inpainting_DPO/logs
git diff > /home/nvme01/H20_Video_inpainting_DPO/logs/pre_videodpo_adapter_dirty_$(date +%Y%m%d_%H%M%S).patch
git stash push -u -m "pre videoinpaint dpo adapter"
git checkout main
git checkout -B h20-videoinpaint-dpo-adapter origin/main
git am /home/nvme01/H20_Video_inpainting_DPO/patches/videodpo/h20_videoinpaint_dpo_adapter.patch

INSTALL_SKIMAGE=1 MAX_PAIR_STEPS=10000 DPO_DIAG_EVERY=300 VAL_EVERY=2000 \
  bash scripts_sh/launch_vc2_dpo_videoinpainting_h20_gpu0_7.sh

LOG=$(ls -t /home/nvme01/H20_Video_inpainting_DPO/logs/vc2_dpo_videoinpainting_h20_gpu2-7_*.stdout.log | head -n 1)
tail -f "$LOG" | grep --line-buffered "\\[dpo_diag\\]\\|\\[video_inpaint_val\\]"
```

All stdout diagnostics, `[dpo_diag]` intermediate metrics, and
`[video_inpaint_val]` PSNR/SSIM lines are written under:

`/home/nvme01/H20_Video_inpainting_DPO/logs`

Step convention:

- `MAX_PAIR_STEPS=10000` means 10000 winner/loser pairs, not 10000 Lightning optimizer steps.
- With the default 6 GPUs and `batch_size=1`, the launch script sets Lightning `max_steps=1667`.
- Logs print both: `[dpo_diag] step=<pair_step>/10000 global_step=<optimizer_step>/1667 ...`.
