# VideoDPO H20 VideoInpainting Adapter

This folder stores a patch for the separate `/home/nvme01/VideoDPO` checkout.
It lets open-source VideoDPO train on the H20 VideoInpainting DPO data root:

`/home/nvme01/H20_Video_inpainting_DPO/DPO_Finetune_Data_Multimodel_v1`

Apply on H20:

```bash
cd /home/nvme01/H20_Video_inpainting_DPO
git pull

cd /home/nvme01/VideoDPO
git status --short
git stash push -u -m "pre videoinpaint dpo adapter"
git checkout main
git checkout -B h20-videoinpaint-dpo-adapter origin/main
git am /home/nvme01/H20_Video_inpainting_DPO/patches/videodpo/h20_videoinpaint_dpo_adapter.patch

bash scripts_sh/launch_vc2_dpo_videoinpainting_h20_gpu6_7.sh
```
