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

bash scripts_sh/launch_vc2_dpo_videoinpainting_h20_gpu6_7.sh

LOG=$(ls -t /home/nvme01/H20_Video_inpainting_DPO/logs/vc2_dpo_videoinpainting_h20_gpu6-7_*.stdout.log | head -n 1)
tail -f "$LOG"
```

All stdout diagnostics, `[dpo_diag]` intermediate metrics, and
`[video_inpaint_val]` PSNR/SSIM lines are written under:

`/home/nvme01/H20_Video_inpainting_DPO/logs`
