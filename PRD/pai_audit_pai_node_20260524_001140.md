# PAI Audit And Asset Preparation

- generated_at: 2026-05-24T00:11:40+08:00
- repo_root: /mnt/nas/hj/H20_Video_inpainting_DPO
- env_file: /mnt/nas/hj/H20_Video_inpainting_DPO/configs/paths/pai.detected.env
- readiness_report: /mnt/nas/hj/H20_Video_inpainting_DPO/PRD/pai_asset_readiness_report.md

## Basic Info
```text
 ____  ______        __
|  _ \/ ___\ \      / /
| | | \___ \\ \ /\ / / 
| |_| |___) |\ V  V /  
|____/|____/  \_/\_/   
                       

Welcome to PAI DSW!

Default python environment: (files in site_packages_path will be committed in image)
╭────────────────────────┬─────────────────────────────────────────┬───────────────────╮
│ path                   │ site_packages_path                      │ commit_to_image   │
├────────────────────────┼─────────────────────────────────────────┼───────────────────┤
│ /usr/local/bin/python3 │ /usr/local/lib/python3.10/site-packages │ yes               │
╰────────────────────────┴─────────────────────────────────────────┴───────────────────╯

Mounted datasets: (mount_path is where you could access dataset in dsw instance, files in mount_path will not be committed in image)
╭────────┬──────────────────────────┬──────────────────────────────┬────────────────────────────────────────────┬───────────────────╮
│ type   │ name                     │ mount_path                   │ dataset_info                               │ commit_to_image   │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ csgo-datasets            │ /mnt/data/csgo-datasets/     │ cpfs-01000vwrt8a6usy68r6wu:/csgo-datasets/ │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ workspace                │ /mnt/workspace               │ cpfs-01000vwrt8a6usy68r6wu:/pku/           │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ pku-workspace            │ /mnt/data/pku/               │ cpfs-01000vwrt8a6usy68r6wu:/pku/           │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ OSS    │ csgo-datasets-oss        │ /mnt/data/csgo-datasets-oss/ │ oss://dataset-csgo-sample-02/shared/       │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ NAS    │ pku-workspace-nas        │ /mnt/nas/                    │ 0329tx2wi8okhqz07m9:/                      │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ csgo-datasets-fullsubset │ /mnt/data/csgo-datasets-     │ cpfs-01000vwrt8a6usy68r6wu:/csgo-datasets- │ no                │
│        │                          │ fullsubset/                  │ fullsubset/                                │                   │
╰────────┴──────────────────────────┴──────────────────────────────┴────────────────────────────────────────────┴───────────────────╯

Sun May 24 12:11:40 AM CST 2026
dsw-753014-dc85766cb-4v2jj
root
/mnt/nas/hj/H20_Video_inpainting_DPO
```

## Git Info
```text
 ____  ______        __
|  _ \/ ___\ \      / /
| | | \___ \\ \ /\ / / 
| |_| |___) |\ V  V /  
|____/|____/  \_/\_/   
                       

Welcome to PAI DSW!

Default python environment: (files in site_packages_path will be committed in image)
╭────────────────────────┬─────────────────────────────────────────┬───────────────────╮
│ path                   │ site_packages_path                      │ commit_to_image   │
├────────────────────────┼─────────────────────────────────────────┼───────────────────┤
│ /usr/local/bin/python3 │ /usr/local/lib/python3.10/site-packages │ yes               │
╰────────────────────────┴─────────────────────────────────────────┴───────────────────╯

Mounted datasets: (mount_path is where you could access dataset in dsw instance, files in mount_path will not be committed in image)
╭────────┬──────────────────────────┬──────────────────────────────┬────────────────────────────────────────────┬───────────────────╮
│ type   │ name                     │ mount_path                   │ dataset_info                               │ commit_to_image   │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ csgo-datasets            │ /mnt/data/csgo-datasets/     │ cpfs-01000vwrt8a6usy68r6wu:/csgo-datasets/ │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ workspace                │ /mnt/workspace               │ cpfs-01000vwrt8a6usy68r6wu:/pku/           │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ pku-workspace            │ /mnt/data/pku/               │ cpfs-01000vwrt8a6usy68r6wu:/pku/           │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ OSS    │ csgo-datasets-oss        │ /mnt/data/csgo-datasets-oss/ │ oss://dataset-csgo-sample-02/shared/       │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ NAS    │ pku-workspace-nas        │ /mnt/nas/                    │ 0329tx2wi8okhqz07m9:/                      │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ csgo-datasets-fullsubset │ /mnt/data/csgo-datasets-     │ cpfs-01000vwrt8a6usy68r6wu:/csgo-datasets- │ no                │
│        │                          │ fullsubset/                  │ fullsubset/                                │                   │
╰────────┴──────────────────────────┴──────────────────────────────┴────────────────────────────────────────────┴───────────────────╯

/mnt/nas/hj/H20_Video_inpainting_DPO
main
 M DPO_finetune/scripts/pai_official_diffueraser_stage.sh
 m external/VBench
 m external/VideoDPO
 M official_videodpo_diffueraser/data.py
 M official_videodpo_diffueraser/models.py
?? .tmp/
?? PRD/pai_asset_readiness_report.md
?? PRD/pai_audit_pai_node_20260523_151346.md
?? PRD/pai_audit_pai_node_20260524_001140.md
?? configs/
?? grit_roi_heads.py
?? scripts/lingbot_process.sh
?? scripts/pai_audit_and_prepare_assets.sh
?? tools/offline_loser_generation.py
076b5ba Support standalone DiffuEraser VAE paths
```

## GPU Info
```text
 ____  ______        __
|  _ \/ ___\ \      / /
| | | \___ \\ \ /\ / / 
| |_| |___) |\ V  V /  
|____/|____/  \_/\_/   
                       

Welcome to PAI DSW!

Default python environment: (files in site_packages_path will be committed in image)
╭────────────────────────┬─────────────────────────────────────────┬───────────────────╮
│ path                   │ site_packages_path                      │ commit_to_image   │
├────────────────────────┼─────────────────────────────────────────┼───────────────────┤
│ /usr/local/bin/python3 │ /usr/local/lib/python3.10/site-packages │ yes               │
╰────────────────────────┴─────────────────────────────────────────┴───────────────────╯

Mounted datasets: (mount_path is where you could access dataset in dsw instance, files in mount_path will not be committed in image)
╭────────┬──────────────────────────┬──────────────────────────────┬────────────────────────────────────────────┬───────────────────╮
│ type   │ name                     │ mount_path                   │ dataset_info                               │ commit_to_image   │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ csgo-datasets            │ /mnt/data/csgo-datasets/     │ cpfs-01000vwrt8a6usy68r6wu:/csgo-datasets/ │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ workspace                │ /mnt/workspace               │ cpfs-01000vwrt8a6usy68r6wu:/pku/           │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ pku-workspace            │ /mnt/data/pku/               │ cpfs-01000vwrt8a6usy68r6wu:/pku/           │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ OSS    │ csgo-datasets-oss        │ /mnt/data/csgo-datasets-oss/ │ oss://dataset-csgo-sample-02/shared/       │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ NAS    │ pku-workspace-nas        │ /mnt/nas/                    │ 0329tx2wi8okhqz07m9:/                      │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ csgo-datasets-fullsubset │ /mnt/data/csgo-datasets-     │ cpfs-01000vwrt8a6usy68r6wu:/csgo-datasets- │ no                │
│        │                          │ fullsubset/                  │ fullsubset/                                │                   │
╰────────┴──────────────────────────┴──────────────────────────────┴────────────────────────────────────────────┴───────────────────╯

Sun May 24 00:11:43 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA L20X                    On  |   00000000:03:00.0 Off |                    0 |
| N/A   30C    P0             74W /  700W |       0MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA L20X                    On  |   00000000:07:00.0 Off |                    0 |
| N/A   30C    P0             77W /  700W |       0MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA L20X                    On  |   00000000:0B:00.0 Off |                    0 |
| N/A   30C    P0             76W /  700W |       0MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA L20X                    On  |   00000000:0F:00.0 Off |                    0 |
| N/A   29C    P0             76W /  700W |       0MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA L20X                    On  |   00000000:14:00.0 Off |                    0 |
| N/A   29C    P0             76W /  700W |     244MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA L20X                    On  |   00000000:18:00.0 Off |                    0 |
| N/A   29C    P0             75W /  700W |       4MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA L20X                    On  |   00000000:1C:00.0 Off |                    0 |
| N/A   30C    P0             76W /  700W |     292MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA L20X                    On  |   00000000:20:00.0 Off |                    0 |
| N/A   32C    P0            123W /  700W |   58071MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

## Python Info
```text
 ____  ______        __
|  _ \/ ___\ \      / /
| | | \___ \\ \ /\ / / 
| |_| |___) |\ V  V /  
|____/|____/  \_/\_/   
                       

Welcome to PAI DSW!

Default python environment: (files in site_packages_path will be committed in image)
╭────────────────────────┬─────────────────────────────────────────┬───────────────────╮
│ path                   │ site_packages_path                      │ commit_to_image   │
├────────────────────────┼─────────────────────────────────────────┼───────────────────┤
│ /usr/local/bin/python3 │ /usr/local/lib/python3.10/site-packages │ yes               │
╰────────────────────────┴─────────────────────────────────────────┴───────────────────╯

Mounted datasets: (mount_path is where you could access dataset in dsw instance, files in mount_path will not be committed in image)
╭────────┬──────────────────────────┬──────────────────────────────┬────────────────────────────────────────────┬───────────────────╮
│ type   │ name                     │ mount_path                   │ dataset_info                               │ commit_to_image   │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ csgo-datasets            │ /mnt/data/csgo-datasets/     │ cpfs-01000vwrt8a6usy68r6wu:/csgo-datasets/ │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ workspace                │ /mnt/workspace               │ cpfs-01000vwrt8a6usy68r6wu:/pku/           │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ pku-workspace            │ /mnt/data/pku/               │ cpfs-01000vwrt8a6usy68r6wu:/pku/           │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ OSS    │ csgo-datasets-oss        │ /mnt/data/csgo-datasets-oss/ │ oss://dataset-csgo-sample-02/shared/       │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ NAS    │ pku-workspace-nas        │ /mnt/nas/                    │ 0329tx2wi8okhqz07m9:/                      │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ csgo-datasets-fullsubset │ /mnt/data/csgo-datasets-     │ cpfs-01000vwrt8a6usy68r6wu:/csgo-datasets- │ no                │
│        │                          │ fullsubset/                  │ fullsubset/                                │                   │
╰────────┴──────────────────────────┴──────────────────────────────┴────────────────────────────────────────────┴───────────────────╯

/usr/local/bin/python
Python 3.10.19
/usr/local/bin/pip
pip 23.0.1 from /usr/local/lib/python3.10/site-packages/pip (python 3.10)
bash: line 1: conda: command not found
```

## Important Python Packages
```text
torch: OK 2.11.0+cu130
torchvision: OK 0.26.0+cu130
diffusers: OK 0.37.1
transformers: OK 5.5.4
accelerate: OK 1.13.0
decord: MISSING or ERROR: No module named 'decord'
cv2: OK 4.13.0
imageio: OK 2.37.3
moviepy: MISSING or ERROR: No module named 'moviepy'
av: MISSING or ERROR: No module named 'av'
numpy: OK 2.4.4
PIL: OK 12.2.0
wandb: MISSING or ERROR: No module named 'wandb'
einops: OK 0.8.2
omegaconf: MISSING or ERROR: No module named 'omegaconf'
```

## Known Completed Experiment Outputs
```text
FOUND /mnt/nas/hj/H20_Video_inpainting_DPO/logs/videodpo_vc2_dpo_official_clean/pai-vc2-dpo-official-full-gpu0-3-gb8-step3000-20260521_061414
FOUND /mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_full/pai-vc2-official-step3000-full-vbench-20260521_141824
FOUND /mnt/nas/hj/H20_Video_inpainting_DPO/logs/official_diffueraser_stage1/pai-official-diffueraser-stage1-full-gpu4-7-gb8-step3000-20260521_072559
FOUND /mnt/nas/hj/H20_Video_inpainting_DPO/logs/official_diffueraser_stage2/pai-official-diffueraser-stage2-full-gpu4-7-gb8-step3000-20260521_150540
FOUND /mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926
FOUND /mnt/nas/hj/H20_Video_inpainting_DPO/logs/qual_sbs_30/vc2_and_diffueraser_20260522
```

## Detected Data And Weight Roots
```text
VIDEO_DPO_OFFICIAL_ROOT=/mnt/nas/hj/official_repos/VideoDPO_official_1febdb4
VIDEO_DPO_DATA_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO/data/videodpo/current
YOUTUBE_VOS_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO/data/youtubevos/current
GENERATED_LOSER_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers
DIFFUERASER_WEIGHT_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO/weights/diffueraser/current
PROPAINTER_WEIGHT_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO/weights/propainter/current
COCOCO_WEIGHT_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO/weights/cococo/current
MINIMAX_REMOVER_WEIGHT_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO/weights/minimax_remover/current
OFFICIAL_VIDEODPO_WEIGHT_ROOT=/mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/checkpoints
VC2_WEIGHT_ROOT=/mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/checkpoints/vc2
```

## Prepared Current Symlinks
- LINKED: /mnt/nas/hj/H20_Video_inpainting_DPO/data/videodpo/current -> /mnt/nas/hj/H20_Video_inpainting_DPO/data/videodpo/current
- LINKED: /mnt/nas/hj/H20_Video_inpainting_DPO/data/youtubevos/current -> /mnt/nas/hj/H20_Video_inpainting_DPO/data/youtubevos/current
- LINKED: /mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/current -> /mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers
- LINKED: /mnt/nas/hj/H20_Video_inpainting_DPO/weights/diffueraser/current -> /mnt/nas/hj/H20_Video_inpainting_DPO/weights/diffueraser/current
- LINKED: /mnt/nas/hj/H20_Video_inpainting_DPO/weights/propainter/current -> /mnt/nas/hj/H20_Video_inpainting_DPO/weights/propainter/current
- LINKED: /mnt/nas/hj/H20_Video_inpainting_DPO/weights/cococo/current -> /mnt/nas/hj/H20_Video_inpainting_DPO/weights/cococo/current
- LINKED: /mnt/nas/hj/H20_Video_inpainting_DPO/weights/minimax_remover/current -> /mnt/nas/hj/H20_Video_inpainting_DPO/weights/minimax_remover/current
- LINKED: /mnt/nas/hj/H20_Video_inpainting_DPO/weights/official_videodpo/current -> /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/checkpoints
- LINKED: /mnt/nas/hj/H20_Video_inpainting_DPO/weights/vc2/current -> /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/checkpoints/vc2

## Four Inpainting Model Search
```text
 ____  ______        __
|  _ \/ ___\ \      / /
| | | \___ \\ \ /\ / / 
| |_| |___) |\ V  V /  
|____/|____/  \_/\_/   
                       

Welcome to PAI DSW!

Default python environment: (files in site_packages_path will be committed in image)
╭────────────────────────┬─────────────────────────────────────────┬───────────────────╮
│ path                   │ site_packages_path                      │ commit_to_image   │
├────────────────────────┼─────────────────────────────────────────┼───────────────────┤
│ /usr/local/bin/python3 │ /usr/local/lib/python3.10/site-packages │ yes               │
╰────────────────────────┴─────────────────────────────────────────┴───────────────────╯

Mounted datasets: (mount_path is where you could access dataset in dsw instance, files in mount_path will not be committed in image)
╭────────┬──────────────────────────┬──────────────────────────────┬────────────────────────────────────────────┬───────────────────╮
│ type   │ name                     │ mount_path                   │ dataset_info                               │ commit_to_image   │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ csgo-datasets            │ /mnt/data/csgo-datasets/     │ cpfs-01000vwrt8a6usy68r6wu:/csgo-datasets/ │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ workspace                │ /mnt/workspace               │ cpfs-01000vwrt8a6usy68r6wu:/pku/           │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ pku-workspace            │ /mnt/data/pku/               │ cpfs-01000vwrt8a6usy68r6wu:/pku/           │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ OSS    │ csgo-datasets-oss        │ /mnt/data/csgo-datasets-oss/ │ oss://dataset-csgo-sample-02/shared/       │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ NAS    │ pku-workspace-nas        │ /mnt/nas/                    │ 0329tx2wi8okhqz07m9:/                      │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ csgo-datasets-fullsubset │ /mnt/data/csgo-datasets-     │ cpfs-01000vwrt8a6usy68r6wu:/csgo-datasets- │ no                │
│        │                          │ fullsubset/                  │ fullsubset/                                │                   │
╰────────┴──────────────────────────┴──────────────────────────────┴────────────────────────────────────────────┴───────────────────╯

./diffueraser
./diffueraser/diffueraser_OR.py
./diffueraser/diffueraser.py
./diffueraser/pipeline_diffueraser.py
./diffueraser/pipeline_diffueraser_stage1.py
./diffueraser/pipeline_diffueraser_stage2.py
./diffueraser/__pycache__/pipeline_diffueraser.cpython-310.pyc
./diffueraser/__pycache__/pipeline_diffueraser_stage1.cpython-310.pyc
./diffueraser/__pycache__/pipeline_diffueraser_stage2.cpython-310.pyc
./DPO_finetune/configs/official_diffueraser_stage1.yaml
./DPO_finetune/configs/official_diffueraser_stage2.yaml
./DPO_finetune/generate_cococo_captions_qwen.py
./DPO_finetune/infer_cococo_candidate.py
./DPO_finetune/infer_diffueraser_candidate.py
./DPO_finetune/infer_minimax_candidate.py
./DPO_finetune/infer_propainter_candidate.py
./DPO_finetune/scripts/generate_cococo_captions_h20.sh
./DPO_finetune/scripts/h20_diffueraser_fullmask_vbench.sh
./DPO_finetune/scripts/pai_official_diffueraser_stage.sh
./DPO_finetune/scripts/sc_videodpo_fullmask_diffueraser_stage1.sbatch
./DPO_finetune/scripts/sc_videodpo_fullmask_diffueraser_stage2.sbatch
./DPO_finetune/VideoDPO_to_DiffuEraser_Report.md
./env_exports/videodpo_hal_diffueraser_compat.environment.yml
./env_exports/videodpo_hal_diffueraser_compat.pip_freeze.txt
./experiments/dpo/stage1/20260516_005803_pai-gpu7-worldmodelphy-fullmask-diffueraser-smoke-20260516_085759
./experiments/dpo/stage1/20260516_010252_pai-gpu7-worldmodelphy-fullmask-diffueraser-smoke-20260516_090248
./experiments/dpo/stage1/20260516_011348_pai-gpu7-worldmodelphy-fullmask-diffueraser-smoke-20260516_091344
./experiments/dpo/stage1/20260516_011857_pai-gpu7-worldmodelphy-fullmask-diffueraser-smoke-20260516_091852
./experiments/dpo/stage1/20260516_013129_lingbot-world-model-fullmask-diffueraser-videodpo-gpu4-7-20260516_093125
./experiments/dpo/stage2/20260520_053908_sc-videodpo-fullmask-diffueraser-stage2
./logs/official_diffueraser_stage1
./logs/official_diffueraser_stage1/pai-official-diffueraser-stage1-full-gpu4-7-gb8-step3000-20260521_072559
./logs/official_diffueraser_stage1/pai-official-diffueraser-stage1-smoke-gpu4-7-20260521_062053
./logs/official_diffueraser_stage1/pai-official-diffueraser-stage1-smoke-gpu4-7-20260521_062939
./logs/official_diffueraser_stage1/pai-official-diffueraser-stage1-smoke-gpu4-7-20260521_065346
./logs/official_diffueraser_stage1/pai-official-diffueraser-stage1-smoke-gpu4-7-20260521_070048
./logs/official_diffueraser_stage1/pai-official-diffueraser-stage1-smoke-gpu4-7-20260521_070612
./logs/official_diffueraser_stage1/pai-official-diffueraser-stage1-smoke-gpu4-7-20260521_070855
./logs/official_diffueraser_stage1/pai-official-diffueraser-stage1-smoke-gpu4-7-20260521_071450
./logs/official_diffueraser_stage2
./logs/official_diffueraser_stage2/pai-official-diffueraser-stage2-full-gpu4-7-gb8-step3000-20260521_145520
./logs/official_diffueraser_stage2/pai-official-diffueraser-stage2-full-gpu4-7-gb8-step3000-20260521_150540
./logs/official_diffueraser_stage2/pai-official-diffueraser-stage2-smoke-gpu4567-20260521_145022
./logs/official_diffueraser_stage2/pai-official-diffueraser-stage2-smoke-gpu4567-20260521_145106
./logs/official_diffueraser_stage2/pai-official-diffueraser-stage2-smoke-gpu4-7-20260521_072409
./logs/pai-official-diffueraser-stage1-full-gpu4-7-gb8-step3000-20260521_072559.log
./logs/pai-official-diffueraser-stage2-full-gpu4-7-gb8-step3000-20260521_150540.log
./logs/qual_sbs_30/vc2_and_diffueraser_20260522
./logs/qual_sbs_30/vc2_and_diffueraser_20260522/diffueraser_base_vs_stage2
./logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_033619
./logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_033619/diffueraser_base
./logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_033619/diffueraser_base.log
./logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_034238
./logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_034238/diffueraser_base
./logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_034238/diffueraser_base.log
./logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_034238/diffueraser_dpo
./logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_034238/diffueraser_dpo.log
./logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_035156
./logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_035156/diffueraser_base
./logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_035156/diffueraser_base.log
./logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_035156/diffueraser_dpo
./logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_035156/diffueraser_dpo.log
./logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926
./logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926/diffueraser_base_full
./logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926/diffueraser_base_qual30
./logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926/diffueraser_stage2
./logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926/logs/diffueraser_stage2_gpu1.log
./logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926/logs/diffueraser_stage2_gpu2.log
./logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926/logs/diffueraser_stage2_gpu3.log
./logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926/logs/diffueraser_stage2_gpu4.log
./logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926/logs/diffueraser_stage2_gpu5.log
./logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926/logs/diffueraser_stage2_gpu6.log
./logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926/run_diffueraser_base_full_vbench_gpus1-6.log
./logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926/run_diffueraser_base_full_vbench_gpus1-6.sh
/mnt/nas/hj/conda_envs/diffueraser
/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter
/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter/ProPainter.pth
/mnt/nas/hj/H20_Video_inpainting_DPO/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO/diffueraser/diffueraser_OR.py
/mnt/nas/hj/H20_Video_inpainting_DPO/diffueraser/diffueraser.py
/mnt/nas/hj/H20_Video_inpainting_DPO/diffueraser/pipeline_diffueraser.py
/mnt/nas/hj/H20_Video_inpainting_DPO/diffueraser/pipeline_diffueraser_stage1.py
/mnt/nas/hj/H20_Video_inpainting_DPO/diffueraser/pipeline_diffueraser_stage2.py
/mnt/nas/hj/H20_Video_inpainting_DPO/diffueraser/__pycache__/pipeline_diffueraser.cpython-310.pyc
/mnt/nas/hj/H20_Video_inpainting_DPO/diffueraser/__pycache__/pipeline_diffueraser_stage1.cpython-310.pyc
/mnt/nas/hj/H20_Video_inpainting_DPO/diffueraser/__pycache__/pipeline_diffueraser_stage2.cpython-310.pyc
/mnt/nas/hj/H20_Video_inpainting_DPO/DPO_finetune/configs/official_diffueraser_stage1.yaml
/mnt/nas/hj/H20_Video_inpainting_DPO/DPO_finetune/configs/official_diffueraser_stage2.yaml
/mnt/nas/hj/H20_Video_inpainting_DPO/DPO_finetune/generate_cococo_captions_qwen.py
/mnt/nas/hj/H20_Video_inpainting_DPO/DPO_finetune/infer_cococo_candidate.py
/mnt/nas/hj/H20_Video_inpainting_DPO/DPO_finetune/infer_diffueraser_candidate.py
/mnt/nas/hj/H20_Video_inpainting_DPO/DPO_finetune/infer_minimax_candidate.py
/mnt/nas/hj/H20_Video_inpainting_DPO/DPO_finetune/infer_propainter_candidate.py
/mnt/nas/hj/H20_Video_inpainting_DPO/DPO_finetune/scripts/generate_cococo_captions_h20.sh
/mnt/nas/hj/H20_Video_inpainting_DPO/DPO_finetune/scripts/h20_diffueraser_fullmask_vbench.sh
/mnt/nas/hj/H20_Video_inpainting_DPO/DPO_finetune/scripts/pai_official_diffueraser_stage.sh
/mnt/nas/hj/H20_Video_inpainting_DPO/DPO_finetune/scripts/sc_videodpo_fullmask_diffueraser_stage1.sbatch
/mnt/nas/hj/H20_Video_inpainting_DPO/DPO_finetune/scripts/sc_videodpo_fullmask_diffueraser_stage2.sbatch
/mnt/nas/hj/H20_Video_inpainting_DPO/DPO_finetune/VideoDPO_to_DiffuEraser_Report.md
/mnt/nas/hj/H20_Video_inpainting_DPO/env_exports/videodpo_hal_diffueraser_compat.environment.yml
/mnt/nas/hj/H20_Video_inpainting_DPO/env_exports/videodpo_hal_diffueraser_compat.pip_freeze.txt
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260516_005803_pai-gpu7-worldmodelphy-fullmask-diffueraser-smoke-20260516_085759
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260516_010252_pai-gpu7-worldmodelphy-fullmask-diffueraser-smoke-20260516_090248
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260516_011348_pai-gpu7-worldmodelphy-fullmask-diffueraser-smoke-20260516_091344
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260516_011857_pai-gpu7-worldmodelphy-fullmask-diffueraser-smoke-20260516_091852
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260516_013129_lingbot-world-model-fullmask-diffueraser-videodpo-gpu4-7-20260516_093125
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260520_053908_sc-videodpo-fullmask-diffueraser-stage2
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/official_diffueraser_stage1
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/official_diffueraser_stage1/pai-official-diffueraser-stage1-full-gpu4-7-gb8-step3000-20260521_072559
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/official_diffueraser_stage1/pai-official-diffueraser-stage1-smoke-gpu4-7-20260521_062053
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/official_diffueraser_stage1/pai-official-diffueraser-stage1-smoke-gpu4-7-20260521_062939
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/official_diffueraser_stage1/pai-official-diffueraser-stage1-smoke-gpu4-7-20260521_065346
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/official_diffueraser_stage1/pai-official-diffueraser-stage1-smoke-gpu4-7-20260521_070048
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/official_diffueraser_stage1/pai-official-diffueraser-stage1-smoke-gpu4-7-20260521_070612
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/official_diffueraser_stage1/pai-official-diffueraser-stage1-smoke-gpu4-7-20260521_070855
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/official_diffueraser_stage1/pai-official-diffueraser-stage1-smoke-gpu4-7-20260521_071450
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/official_diffueraser_stage2
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/official_diffueraser_stage2/pai-official-diffueraser-stage2-full-gpu4-7-gb8-step3000-20260521_145520
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/official_diffueraser_stage2/pai-official-diffueraser-stage2-full-gpu4-7-gb8-step3000-20260521_150540
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/official_diffueraser_stage2/pai-official-diffueraser-stage2-smoke-gpu4567-20260521_145022
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/official_diffueraser_stage2/pai-official-diffueraser-stage2-smoke-gpu4567-20260521_145106
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/official_diffueraser_stage2/pai-official-diffueraser-stage2-smoke-gpu4-7-20260521_072409
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/pai-official-diffueraser-stage1-full-gpu4-7-gb8-step3000-20260521_072559.log
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/pai-official-diffueraser-stage2-full-gpu4-7-gb8-step3000-20260521_150540.log
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/qual_sbs_30/vc2_and_diffueraser_20260522
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/qual_sbs_30/vc2_and_diffueraser_20260522/diffueraser_base_vs_stage2
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_033619
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_033619/diffueraser_base
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_033619/diffueraser_base.log
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_034238
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_034238/diffueraser_base
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_034238/diffueraser_base.log
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_034238/diffueraser_dpo
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_034238/diffueraser_dpo.log
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_035156
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_035156/diffueraser_base
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_035156/diffueraser_base.log
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_035156/diffueraser_dpo
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_035156/diffueraser_dpo.log
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926/diffueraser_base_full
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926/diffueraser_base_qual30
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926/diffueraser_stage2
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926/logs/diffueraser_stage2_gpu1.log
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926/logs/diffueraser_stage2_gpu2.log
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926/logs/diffueraser_stage2_gpu3.log
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926/logs/diffueraser_stage2_gpu4.log
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926/logs/diffueraser_stage2_gpu5.log
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926/logs/diffueraser_stage2_gpu6.log
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926/run_diffueraser_base_full_vbench_gpus1-6.log
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926/run_diffueraser_base_full_vbench_gpus1-6.sh
/mnt/nas/hj/H20_Video_inpainting_DPO/official_videodpo_diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO/propainter/model/propainter.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/diffueraser/diffueraser_OR.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/diffueraser/diffueraser.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/diffueraser/pipeline_diffueraser.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/diffueraser/pipeline_diffueraser_stage1.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/diffueraser/pipeline_diffueraser_stage2.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/diffueraser/__pycache__/diffueraser_OR.cpython-310.pyc
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/diffueraser/__pycache__/pipeline_diffueraser.cpython-310.pyc
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/diffueraser/__pycache__/pipeline_diffueraser_stage1.cpython-310.pyc
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_finetune/captions/cococo_qwen_smoke_captions.json
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_17c220e4f6_mask20266422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_17c220e4f6_mask20266422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_17c220e4f6_mask20266422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_17c220e4f6_mask20266422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_17c220e4f6_mask20266423/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_17c220e4f6_mask20266423/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_17c220e4f6_mask20266423/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_17c220e4f6_mask20266423/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_1de4a9e537_mask20272422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_1de4a9e537_mask20272422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_1de4a9e537_mask20272422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_1de4a9e537_mask20272422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_1de4a9e537_mask20272423/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_1de4a9e537_mask20272423/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_1de4a9e537_mask20272423/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_1de4a9e537_mask20272423/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_2f53998171_mask20277422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_2f53998171_mask20277422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_2f53998171_mask20277422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_2f53998171_mask20277422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_2f53998171_mask20277423/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_2f53998171_mask20277423/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_2f53998171_mask20277423/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_2f53998171_mask20277423/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_343bc6a65a_mask20271422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_343bc6a65a_mask20271422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_343bc6a65a_mask20271422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_343bc6a65a_mask20271422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_343bc6a65a_mask20271423/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_343bc6a65a_mask20271423/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_343bc6a65a_mask20271423/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_343bc6a65a_mask20271423/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_42f5c61c49_mask20261422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_42f5c61c49_mask20261422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_42f5c61c49_mask20261422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_42f5c61c49_mask20261422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_42f5c61c49_mask20261423/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_42f5c61c49_mask20261423/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_42f5c61c49_mask20261423/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_42f5c61c49_mask20261423/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_4e70279712_mask20263422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_4e70279712_mask20263422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_4e70279712_mask20263422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_4e70279712_mask20263422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_4e70279712_mask20263423/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_4e70279712_mask20263423/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_4e70279712_mask20263423/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_4e70279712_mask20263423/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_5678a91bd8_mask20264422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_5678a91bd8_mask20264422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_5678a91bd8_mask20264422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_5678a91bd8_mask20264422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_5678a91bd8_mask20264423/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_5678a91bd8_mask20264423/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_5678a91bd8_mask20264423/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_5678a91bd8_mask20264423/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_579393912d_mask20273422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_579393912d_mask20273422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_579393912d_mask20273422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_579393912d_mask20273422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_579393912d_mask20273423/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_579393912d_mask20273423/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_579393912d_mask20273423/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_579393912d_mask20273423/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_7163b8085f_mask20276422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_7163b8085f_mask20276422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_7163b8085f_mask20276422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_7163b8085f_mask20276422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_7163b8085f_mask20276423/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_7163b8085f_mask20276423/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_7163b8085f_mask20276423/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_7163b8085f_mask20276423/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_7f5faedf8b_mask20269422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_7f5faedf8b_mask20269422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_7f5faedf8b_mask20269422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_7f5faedf8b_mask20269422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_7f5faedf8b_mask20269423/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_7f5faedf8b_mask20269423/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_7f5faedf8b_mask20269423/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_7f5faedf8b_mask20269423/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_85b1eda0d9_mask20267422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_85b1eda0d9_mask20267422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_85b1eda0d9_mask20267422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_85b1eda0d9_mask20267422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_85b1eda0d9_mask20267423/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_85b1eda0d9_mask20267423/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_85b1eda0d9_mask20267423/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_85b1eda0d9_mask20267423/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_a6f4e0817f_mask20275422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_a6f4e0817f_mask20275422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_a6f4e0817f_mask20275422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_a6f4e0817f_mask20275422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_a6f4e0817f_mask20275423/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_a6f4e0817f_mask20275423/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_a6f4e0817f_mask20275423/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_a6f4e0817f_mask20275423/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_a832fa8790_mask20274422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_a832fa8790_mask20274422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_a832fa8790_mask20274422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_a832fa8790_mask20274422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_a832fa8790_mask20274423/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_a832fa8790_mask20274423/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_a832fa8790_mask20274423/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_a832fa8790_mask20274423/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_aab33f0e2a_mask20280422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_aab33f0e2a_mask20280422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_aab33f0e2a_mask20280422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_aab33f0e2a_mask20280422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_aab33f0e2a_mask20280423/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_aab33f0e2a_mask20280423/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_aab33f0e2a_mask20280423/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_aab33f0e2a_mask20280423/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_aae78feda4_mask20265422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_aae78feda4_mask20265422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_aae78feda4_mask20265422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_aae78feda4_mask20265422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_aae78feda4_mask20265423/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_aae78feda4_mask20265423/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_aae78feda4_mask20265423/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_aae78feda4_mask20265423/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_b9f43ef41e_mask20279422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_b9f43ef41e_mask20279422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_b9f43ef41e_mask20279422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_b9f43ef41e_mask20279422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_b9f43ef41e_mask20279423/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_b9f43ef41e_mask20279423/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_b9f43ef41e_mask20279423/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_b9f43ef41e_mask20279423/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_bdb74e333f_mask20262422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_bdb74e333f_mask20262422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_bdb74e333f_mask20262422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_bdb74e333f_mask20262422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_bdb74e333f_mask20262423/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_bdb74e333f_mask20262423/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_bdb74e333f_mask20262423/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_bdb74e333f_mask20262423/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_be9b29e08e_mask20270422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_be9b29e08e_mask20270422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_be9b29e08e_mask20270422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_be9b29e08e_mask20270422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_be9b29e08e_mask20270423/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_be9b29e08e_mask20270423/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_be9b29e08e_mask20270423/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_be9b29e08e_mask20270423/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_c3dd38bf98_mask20268422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_c3dd38bf98_mask20268422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_c3dd38bf98_mask20268422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_c3dd38bf98_mask20268422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_c3dd38bf98_mask20268423/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_c3dd38bf98_mask20268423/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_c3dd38bf98_mask20268423/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_c3dd38bf98_mask20268423/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_ce1bc5743a_mask20278422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_ce1bc5743a_mask20278422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_ce1bc5743a_mask20278422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_ce1bc5743a_mask20278422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_ce1bc5743a_mask20278423/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_ce1bc5743a_mask20278423/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_ce1bc5743a_mask20278423/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_debug_keep_logs_20260425_074508/ytbv_ce1bc5743a_mask20278423/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_v1.manifest.before_short_diffueraser_repair_1777078569.json
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_v1/_repair_short_diffueraser_ids.txt
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_v1.repair_short_diffueraser.stdout.log
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_v1/ytbv_6353f09384_mask21145423/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_v1/ytbv_6353f09384_mask21145423/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_v1/ytbv_7613df1f84_mask21144423/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_v1/ytbv_7613df1f84_mask21144423/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_Finetune_Data_Multimodel_v1/ytbv_7613df1f84_mask21144423/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_finetune/generate_cococo_captions_qwen.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_finetune/infer_cococo_candidate.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_finetune/infer_diffueraser_candidate.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_finetune/infer_minimax_candidate.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_finetune/infer_propainter_candidate.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_finetune/__pycache__/infer_diffueraser_candidate.cpython-313.pyc
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_finetune/scripts/generate_cococo_captions_h20.sh
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_finetune/scripts/sc_videodpo_fullmask_diffueraser_stage1.sbatch
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/DPO_finetune/VideoDPO_to_DiffuEraser_Report.md
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/env_exports/videodpo_hal_diffueraser_compat.environment.yml
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/env_exports/videodpo_hal_diffueraser_compat.pip_freeze.txt
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/experiments/dpo/stage1/20260513_194542_h20-gpu7-videodpo-fullmask-diffueraser-smoke-20260514_034541
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/propainter/model/propainter.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/propainter/model/__pycache__/propainter.cpython-310.pyc
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_MaskPolicy_Check_20260423_183737/ytbv_42f5c61c49_mask20261422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_MaskPolicy_Check_20260423_183737/ytbv_42f5c61c49_mask20261423/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_MaskPolicy_Check_20260423_183737/ytbv_bdb74e333f_mask20262422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_MaskPolicy_Check_20260423_183737/ytbv_bdb74e333f_mask20262423/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Metric_Check_20260423_185835/ytbv_42f5c61c49_mask20261422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Metric_Check_20260423_185835/ytbv_42f5c61c49_mask20261422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Metric_Check_20260423_185835/ytbv_42f5c61c49_mask20261422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Metric_Check_20260423_185835/ytbv_42f5c61c49_mask20261422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Metric_Check_20260423_190452/ytbv_42f5c61c49_mask20261422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Metric_Check_20260423_190452/ytbv_42f5c61c49_mask20261422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Metric_Check_20260423_190452/ytbv_42f5c61c49_mask20261422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Metric_Check_20260423_190452/ytbv_42f5c61c49_mask20261422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Metric_Check_20260423_191204/ytbv_42f5c61c49_mask20261422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Metric_Check_20260423_191204/ytbv_42f5c61c49_mask20261422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Metric_Check_20260423_191204/ytbv_42f5c61c49_mask20261422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Metric_Check_20260423_191204/ytbv_42f5c61c49_mask20261422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Metric_Check_20260423_191706/ytbv_42f5c61c49_mask20261422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Metric_Check_20260423_191706/ytbv_42f5c61c49_mask20261422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Metric_Check_20260423_191706/ytbv_42f5c61c49_mask20261422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Metric_Check_20260423_191706/ytbv_42f5c61c49_mask20261422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Metric_Check_20260423_192132/ytbv_42f5c61c49_mask20261422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Metric_Check_20260423_192132/ytbv_42f5c61c49_mask20261422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Metric_Check_20260423_192132/ytbv_42f5c61c49_mask20261422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Metric_Check_20260423_192132/ytbv_42f5c61c49_mask20261422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_172356/ytbv_42f5c61c49_mask20261422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_172356/ytbv_42f5c61c49_mask20261422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_172356/ytbv_42f5c61c49_mask20261422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_172356/ytbv_42f5c61c49_mask20261422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_172356/ytbv_bdb74e333f_mask20262422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_172356/ytbv_bdb74e333f_mask20262422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_172356/ytbv_bdb74e333f_mask20262422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_172356/ytbv_bdb74e333f_mask20262422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_172959/ytbv_42f5c61c49_mask20261422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_172959/ytbv_42f5c61c49_mask20261422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_172959/ytbv_42f5c61c49_mask20261422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_172959/ytbv_bdb74e333f_mask20262422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_172959/ytbv_bdb74e333f_mask20262422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_172959/ytbv_bdb74e333f_mask20262422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_173017/ytbv_42f5c61c49_mask20261422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_173017/ytbv_42f5c61c49_mask20261422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_173017/ytbv_42f5c61c49_mask20261422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_173017/ytbv_bdb74e333f_mask20262422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_173017/ytbv_bdb74e333f_mask20262422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_173017/ytbv_bdb74e333f_mask20262422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_173850/ytbv_42f5c61c49_mask20261422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_173850/ytbv_42f5c61c49_mask20261422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_173850/ytbv_42f5c61c49_mask20261422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_173850/ytbv_bdb74e333f_mask20262422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_173850/ytbv_bdb74e333f_mask20262422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_173850/ytbv_bdb74e333f_mask20262422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_175919/ytbv_42f5c61c49_mask20261422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_175919/ytbv_42f5c61c49_mask20261422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_175919/ytbv_bdb74e333f_mask20262422/candidates/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_Multimodel_Smoke_20260423_175919/ytbv_bdb74e333f_mask20262422/candidates/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_17c220e4f6_mask20266422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_17c220e4f6_mask20266422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_1de4a9e537_mask20272422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_1de4a9e537_mask20272422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_2f53998171_mask20277422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_2f53998171_mask20277422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_343bc6a65a_mask20271422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_343bc6a65a_mask20271422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_42f5c61c49_mask20261422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_42f5c61c49_mask20261422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_4e70279712_mask20263422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_4e70279712_mask20263422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_5678a91bd8_mask20264422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_5678a91bd8_mask20264422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_579393912d_mask20273422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_579393912d_mask20273422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_7163b8085f_mask20276422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_7163b8085f_mask20276422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_7f5faedf8b_mask20269422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_7f5faedf8b_mask20269422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_85b1eda0d9_mask20267422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_85b1eda0d9_mask20267422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_a6f4e0817f_mask20275422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_a6f4e0817f_mask20275422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_a832fa8790_mask20274422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_a832fa8790_mask20274422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_aab33f0e2a_mask20280422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_aab33f0e2a_mask20280422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_aae78feda4_mask20265422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_aae78feda4_mask20265422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_b9f43ef41e_mask20279422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_b9f43ef41e_mask20279422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_bdb74e333f_mask20262422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_bdb74e333f_mask20262422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_be9b29e08e_mask20270422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_be9b29e08e_mask20270422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_c3dd38bf98_mask20268422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_c3dd38bf98_mask20268422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_ce1bc5743a_mask20278422/candidates/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/DPO_NewBand20_gpu01_no_vbench_20260429_091502/ytbv_ce1bc5743a_mask20278422/candidates/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/five_panel_smoke_comparisons_h264/ytbv_42f5c61c49_mask20261422_gt_mask_propainter_cococo_diffueraser_minimax.mp4
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/five_panel_smoke_comparisons_h264/ytbv_bdb74e333f_mask20262422_gt_mask_propainter_cococo_diffueraser_minimax.mp4
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/five_panel_smoke_comparisons/ytbv_42f5c61c49_mask20261422_gt_mask_propainter_cococo_diffueraser_minimax.mp4
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/smoke_outputs/five_panel_smoke_comparisons/ytbv_bdb74e333f_mask20262422_gt_mask_propainter_cococo_diffueraser_minimax.mp4
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/downloads/cococo_hf_extract
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/envs/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/envs/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/envs/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/manifests/diffueraser_h20_env.yml
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/manifests/diffueraser_h20_pip_freeze.txt
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/COCOCO
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/COCOCO/__asset__/COCOCO.PNG
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/COCOCO/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/DiffuEraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/DiffuEraser/assets/DiffuEraser_pipeline.png
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/DiffuEraser/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/DiffuEraser/diffueraser/diffueraser.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/DiffuEraser/diffueraser/pipeline_diffueraser.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/DiffuEraser/diffueraser/pipeline_diffueraser_stage1.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/DiffuEraser/diffueraser/pipeline_diffueraser_stage2.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/DiffuEraser/eval_DiffuEraser_stage1.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/DiffuEraser/eval_DiffuEraser_stage2.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/DiffuEraser/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/DiffuEraser/run_diffueraser.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/DiffuEraser/train_DiffuEraser_stage1.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/DiffuEraser/train_DiffuEraser_stage1.sh
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/DiffuEraser/train_DiffuEraser_stage2.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/DiffuEraser/train_DiffuEraser_stage2.sh
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/MiniMax-Remover
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/MiniMax-Remover/gradio_demo/pipeline_minimax_remover.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/MiniMax-Remover/gradio_demo/transformer_minimax_remover.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/MiniMax-Remover/pipeline_minimax_remover.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/MiniMax-Remover/__pycache__/pipeline_minimax_remover.cpython-310.pyc
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/MiniMax-Remover/__pycache__/transformer_minimax_remover.cpython-310.pyc
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/MiniMax-Remover/test_minimax_remover.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/MiniMax-Remover/transformer_minimax_remover.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/ProPainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/ProPainter/assets/propainter_logo1_glow.png
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/ProPainter/assets/propainter_logo1.png
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/ProPainter/assets/ProPainter_pipeline.png
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/ProPainter/configs/train_propainter.json
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/ProPainter/inference_propainter.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/ProPainter/model/propainter.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/ProPainter/scripts/evaluate_propainter.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/COCOCO_weight
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/COCOCO_weight/cococo
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/diffueraser/Orign_Diffueraser
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/minimax
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/propainter
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/propainter/ProPainter.pth
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/tools/generate_diffueraser_fullmask_vbench.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/train_DiffuEraser_stage1.py
/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/train_DiffuEraser_stage2.py
/mnt/nas/hj/H20_Video_inpainting_DPO/.tmp/pre_pull_adapter_20260521_062926/DPO_finetune/configs/official_diffueraser_stage1.yaml
/mnt/nas/hj/H20_Video_inpainting_DPO/.tmp/pre_pull_adapter_20260521_062926/DPO_finetune/configs/official_diffueraser_stage2.yaml
/mnt/nas/hj/H20_Video_inpainting_DPO/.tmp/pre_pull_adapter_20260521_062926/DPO_finetune/scripts/pai_official_diffueraser_stage.sh
/mnt/nas/hj/H20_Video_inpainting_DPO/.tmp/pre_pull_adapter_20260521_062926/official_videodpo_diffueraser
```

## Generation Script Search
```text
 ____  ______        __
|  _ \/ ___\ \      / /
| | | \___ \\ \ /\ / / 
| |_| |___) |\ V  V /  
|____/|____/  \_/\_/   
                       

Welcome to PAI DSW!

Default python environment: (files in site_packages_path will be committed in image)
╭────────────────────────┬─────────────────────────────────────────┬───────────────────╮
│ path                   │ site_packages_path                      │ commit_to_image   │
├────────────────────────┼─────────────────────────────────────────┼───────────────────┤
│ /usr/local/bin/python3 │ /usr/local/lib/python3.10/site-packages │ yes               │
╰────────────────────────┴─────────────────────────────────────────┴───────────────────╯

Mounted datasets: (mount_path is where you could access dataset in dsw instance, files in mount_path will not be committed in image)
╭────────┬──────────────────────────┬──────────────────────────────┬────────────────────────────────────────────┬───────────────────╮
│ type   │ name                     │ mount_path                   │ dataset_info                               │ commit_to_image   │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ csgo-datasets            │ /mnt/data/csgo-datasets/     │ cpfs-01000vwrt8a6usy68r6wu:/csgo-datasets/ │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ workspace                │ /mnt/workspace               │ cpfs-01000vwrt8a6usy68r6wu:/pku/           │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ pku-workspace            │ /mnt/data/pku/               │ cpfs-01000vwrt8a6usy68r6wu:/pku/           │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ OSS    │ csgo-datasets-oss        │ /mnt/data/csgo-datasets-oss/ │ oss://dataset-csgo-sample-02/shared/       │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ NAS    │ pku-workspace-nas        │ /mnt/nas/                    │ 0329tx2wi8okhqz07m9:/                      │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ csgo-datasets-fullsubset │ /mnt/data/csgo-datasets-     │ cpfs-01000vwrt8a6usy68r6wu:/csgo-datasets- │ no                │
│        │                          │ fullsubset/                  │ fullsubset/                                │                   │
╰────────┴──────────────────────────┴──────────────────────────────┴────────────────────────────────────────────┴───────────────────╯

./diffueraser/diffueraser_OR.py
./diffueraser/diffueraser.py
./diffueraser/__init__.py
./diffueraser/metrics.py
./diffueraser/patch_mask_morph_v2.py
./diffueraser/pipeline_diffueraser.py
./diffueraser/pipeline_diffueraser_stage1.py
./diffueraser/pipeline_diffueraser_stage2.py
./DPO_finetune/generate_cococo_captions_qwen.py
./DPO_finetune/generate_multimodel_dpo_dataset.py
./DPO_finetune/infer_cococo_candidate.py
./DPO_finetune/infer_diffueraser_candidate.py
./DPO_finetune/infer_minimax_candidate.py
./DPO_finetune/infer_propainter_candidate.py
./DPO_finetune/scripts/04_generate_multimodel_dpo_newband.sbatch
./DPO_finetune/scripts/generate_cococo_captions_h20.sh
./DPO_finetune/scripts/h20_diffueraser_fullmask_vbench.sh
./DPO_finetune/scripts/pai_official_diffueraser_stage.sh
./DPO_finetune/scripts/sc_videodpo_fullmask_diffueraser_stage1.sbatch
./DPO_finetune/scripts/sc_videodpo_fullmask_diffueraser_stage2.sbatch
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/custom/infer.sh
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/infer/cli_demo.sh
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/infer/demo_agent.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/infer/demo_bert.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/infer/demo_grounding.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/infer/demo_hf.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/infer/demo_lora.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/infer/demo_mllm.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/infer/demo.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/infer/demo_reward_model.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/infer/lmdeploy/mllm_tp.sh
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/infer/pt/batch_ddp.sh
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/infer/pt/bert.sh
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/infer/pt/lora.sh
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/infer/pt/mllm_device_map.sh
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/infer/pt/prm.sh
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/infer/pt/reward_model.sh
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/infer/vllm/dp_tp.sh
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/infer/vllm/mllm_ddp.sh
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/infer/vllm/mllm_tp.sh
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/notebook/qwen2_5-self-cognition/infer.sh
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/train/agent/loss_scale/infer_lora.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/train/all_to_all/infer.sh
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/train/full/infer.sh
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/train/grpo/qwen2_5_omni/infer.sh
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/train/infer.sh
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/train/multimodal/infer.sh
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/train/multimodal/lora_llm_full_vit/infer.sh
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/train/multimodal/omni/infer.sh
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/train/predict_with_generate/train.sh
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/train/seq_cls/bert/infer.sh
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/train/seq_cls/qwen2_5/infer.sh
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/train/seq_cls/qwen2_vl/infer.sh
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/examples/train/seq_cls/regression/infer.sh
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/scripts/benchmark/generate_report.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/swift/cli/infer.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/swift/llm/argument/infer_args.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/swift/llm/infer/deploy.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/swift/llm/infer/infer_engine/base.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/swift/llm/infer/infer_engine/grpo_vllm_engine.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/swift/llm/infer/infer_engine/infer_client.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/swift/llm/infer/infer_engine/infer_engine.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/swift/llm/infer/infer_engine/__init__.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/swift/llm/infer/infer_engine/lmdeploy_engine.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/swift/llm/infer/infer_engine/patch.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/swift/llm/infer/infer_engine/pt_engine.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/swift/llm/infer/infer_engine/utils.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/swift/llm/infer/infer_engine/vllm_engine.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/swift/llm/infer/infer.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/swift/llm/infer/__init__.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/swift/llm/infer/protocol.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/swift/llm/infer/rollout.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/swift/llm/infer/utils.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/swift/llm/model/model/minimax.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/swift/llm/template/template/minimax.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/swift/ui/llm_infer/generate.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/swift/ui/llm_infer/__init__.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/swift/ui/llm_infer/llm_infer.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/swift/ui/llm_infer/model.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/swift/ui/llm_infer/runtime.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/tests/infer/test_agent.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/tests/infer/test_infer.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/tests/infer/test_logprobs.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/tests/infer/test_main.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/tests/infer/test_max_memory.py
./external/VBench/VBench-2.0/vbench2/third_party/Instance_detector/tests/infer/test_mllm.py
./external/VBench/VBench-2.0/vbench2/third_party/LLaVA_NeXT/playground/sgl_llava_inference_multinode.py
./external/VBench/VBench-2.0/vbench2/third_party/ViTDetector/inference.py
./external/VBench/VBench-2.0/vbench2/third_party/ViTDetector/third_party/YOLO-World/tools/generate_image_prompts.py
./external/VBench/VBench-2.0/vbench2/third_party/ViTDetector/third_party/YOLO-World/tools/generate_text_prompts.py
./external/VBench/VBench-2.0/vbench2/third_party/YOLO-World/tools/generate_image_prompts.py
./external/VBench/VBench-2.0/vbench2/third_party/YOLO-World/tools/generate_text_prompts.py
./external/VideoDPO/scripts/inference_ddp.py
./external/VideoDPO/scripts/inference.py
./external/VideoDPO/scripts/inference_utils.py
./external/VideoDPO/scripts_sh/inference_t2v.sh
./external/VideoDPO/scripts/turbo_inference/text2video.py
./external/VideoDPO/scripts/turbo_inference/vbench_videos.py
./inference/compare_all.py
./inference/generate_captions_BR.py
./inference/generate_captions_OR.py
./inference/generate_report.py
./inference/__init__.py
./inference/metrics.py
./inference/run_20exp_unified.sh
./inference/run_BR.py
./inference/run_OR.py
./inference/run_weight_sweep.sh
./inference/start_weight_sweep.sh
./logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926/generate_selected_base_pairs.py
./logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926/run_diffueraser_base_full_vbench_gpus1-6.sh
./logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926/run_fullmask_multigpu.sh
./logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926/run_fullmask_multigpu_v2_base30.sh
./official_videodpo_diffueraser/data.py
./official_videodpo_diffueraser/__init__.py
./official_videodpo_diffueraser/models.py
./propainter/core/dataset.py
./propainter/core/dist.py
./propainter/core/loss.py
./propainter/core/lr_scheduler.py
./propainter/core/metrics.py
./propainter/core/prefetch_dataloader.py
./propainter/core/trainer_flow_w_edge.py
./propainter/core/trainer.py
./propainter/core/utils.py
./propainter/inference_OR.py
./propainter/inference.py
./propainter/__init__.py
./propainter/model/canny/canny_filter.py
./propainter/model/canny/filter.py
./propainter/model/canny/gaussian.py
./propainter/model/canny/kernels.py
./propainter/model/canny/sobel.py
./propainter/model/__init__.py
./propainter/model/misc.py
./propainter/model/modules/base_module.py
./propainter/model/modules/deformconv.py
./propainter/model/modules/flow_comp_raft.py
./propainter/model/modules/flow_loss_utils.py
./propainter/model/modules/sparse_transformer.py
./propainter/model/modules/spectral_norm.py
./propainter/model/propainter.py
./propainter/model/recurrent_flow_completion.py
./propainter/model/vgg_arch.py
./propainter/RAFT/corr.py
./propainter/RAFT/datasets.py
./propainter/RAFT/demo.py
./propainter/RAFT/extractor.py
./propainter/RAFT/__init__.py
./propainter/RAFT/raft.py
./propainter/RAFT/update.py
./propainter/RAFT/utils/augmentor.py
./propainter/RAFT/utils/flow_viz_pt.py
./propainter/RAFT/utils/flow_viz.py
./propainter/RAFT/utils/frame_utils.py
./propainter/RAFT/utils/__init__.py
./propainter/RAFT/utils/utils.py
./propainter/utils/download_util.py
./propainter/utils/file_client.py
./propainter/utils/flow_util.py
./propainter/utils/img_util.py
./.tmp/pre_pull_adapter_20260521_062926/DPO_finetune/scripts/pai_official_diffueraser_stage.sh
./.tmp/pre_pull_adapter_20260521_062926/official_videodpo_diffueraser/data.py
./.tmp/pre_pull_adapter_20260521_062926/official_videodpo_diffueraser/__init__.py
./.tmp/pre_pull_adapter_20260521_062926/official_videodpo_diffueraser/models.py
./tools/generate_diffueraser_fullmask_vbench.py
./tools/score_inpainting_quality.py
./train_DiffuEraser_stage1.py
./train_DiffuEraser_stage2.py
```

## Large Asset Check
```text
 ____  ______        __
|  _ \/ ___\ \      / /
| | | \___ \\ \ /\ / / 
| |_| |___) |\ V  V /  
|____/|____/  \_/\_/   
                       

Welcome to PAI DSW!

Default python environment: (files in site_packages_path will be committed in image)
╭────────────────────────┬─────────────────────────────────────────┬───────────────────╮
│ path                   │ site_packages_path                      │ commit_to_image   │
├────────────────────────┼─────────────────────────────────────────┼───────────────────┤
│ /usr/local/bin/python3 │ /usr/local/lib/python3.10/site-packages │ yes               │
╰────────────────────────┴─────────────────────────────────────────┴───────────────────╯

Mounted datasets: (mount_path is where you could access dataset in dsw instance, files in mount_path will not be committed in image)
╭────────┬──────────────────────────┬──────────────────────────────┬────────────────────────────────────────────┬───────────────────╮
│ type   │ name                     │ mount_path                   │ dataset_info                               │ commit_to_image   │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ csgo-datasets            │ /mnt/data/csgo-datasets/     │ cpfs-01000vwrt8a6usy68r6wu:/csgo-datasets/ │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ workspace                │ /mnt/workspace               │ cpfs-01000vwrt8a6usy68r6wu:/pku/           │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ pku-workspace            │ /mnt/data/pku/               │ cpfs-01000vwrt8a6usy68r6wu:/pku/           │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ OSS    │ csgo-datasets-oss        │ /mnt/data/csgo-datasets-oss/ │ oss://dataset-csgo-sample-02/shared/       │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ NAS    │ pku-workspace-nas        │ /mnt/nas/                    │ 0329tx2wi8okhqz07m9:/                      │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ csgo-datasets-fullsubset │ /mnt/data/csgo-datasets-     │ cpfs-01000vwrt8a6usy68r6wu:/csgo-datasets- │ no                │
│        │                          │ fullsubset/                  │ fullsubset/                                │                   │
╰────────┴──────────────────────────┴──────────────────────────────┴────────────────────────────────────────────┴───────────────────╯

```

## Disk Capacity
```text
 ____  ______        __
|  _ \/ ___\ \      / /
| | | \___ \\ \ /\ / / 
| |_| |___) |\ V  V /  
|____/|____/  \_/\_/   
                       

Welcome to PAI DSW!

Default python environment: (files in site_packages_path will be committed in image)
╭────────────────────────┬─────────────────────────────────────────┬───────────────────╮
│ path                   │ site_packages_path                      │ commit_to_image   │
├────────────────────────┼─────────────────────────────────────────┼───────────────────┤
│ /usr/local/bin/python3 │ /usr/local/lib/python3.10/site-packages │ yes               │
╰────────────────────────┴─────────────────────────────────────────┴───────────────────╯

Mounted datasets: (mount_path is where you could access dataset in dsw instance, files in mount_path will not be committed in image)
╭────────┬──────────────────────────┬──────────────────────────────┬────────────────────────────────────────────┬───────────────────╮
│ type   │ name                     │ mount_path                   │ dataset_info                               │ commit_to_image   │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ csgo-datasets            │ /mnt/data/csgo-datasets/     │ cpfs-01000vwrt8a6usy68r6wu:/csgo-datasets/ │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ workspace                │ /mnt/workspace               │ cpfs-01000vwrt8a6usy68r6wu:/pku/           │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ pku-workspace            │ /mnt/data/pku/               │ cpfs-01000vwrt8a6usy68r6wu:/pku/           │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ OSS    │ csgo-datasets-oss        │ /mnt/data/csgo-datasets-oss/ │ oss://dataset-csgo-sample-02/shared/       │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ NAS    │ pku-workspace-nas        │ /mnt/nas/                    │ 0329tx2wi8okhqz07m9:/                      │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ csgo-datasets-fullsubset │ /mnt/data/csgo-datasets-     │ cpfs-01000vwrt8a6usy68r6wu:/csgo-datasets- │ no                │
│        │                          │ fullsubset/                  │ fullsubset/                                │                   │
╰────────┴──────────────────────────┴──────────────────────────────┴────────────────────────────────────────────┴───────────────────╯

Filesystem                                                                                                Size  Used Avail Use% Mounted on
40b4a7ead0769f65bd0322230f875073e9bdb9d4b6ff190678cd1e19b73e14b9-rootfs                                   5.3T   19G  5.0T   1% /
tmpfs                                                                                                      64M     0   64M   0% /dev
tmpfs                                                                                                     931G     0  931G   0% /sys/fs/cgroup
/dev/vda                                                                                                   30G  1.2G   27G   5% /tmp
virtiofs-default                                                                                          7.0T  358G  6.3T   6% /etc/dsw/config
overlay                                                                                                    30G  1.2G   27G   5% /etc/dsw
tmpfs                                                                                                     1.8T  3.2G  1.8T   1% /dev/shm
rund:TS9haq7t:cpfs-01000vwrt8a6usy68r6wu-000001.cn-shanghai.cpfs.aliyuncs.com:/pku/                        70T   64T  6.8T  91% /mnt/data/pku
tmpfs                                                                                                     931G  4.1M  931G   1% /etc/hosts
172.28.48.25:/                                                                                             10P  3.7T   10P   1% /mnt/nas
rund:yyMeb9C3:cpfs-01000vwrt8a6usy68r6wu-000001.cn-shanghai.cpfs.aliyuncs.com:/csgo-datasets-fullsubset/   70T   64T  6.8T  91% /mnt/data/csgo-datasets-fullsubset
ossfs2                                                                                                    512T     0  512T   0% /mnt/data/csgo-datasets-oss
rund:6FBaGknW:cpfs-01000vwrt8a6usy68r6wu-000001.cn-shanghai.cpfs.aliyuncs.com:/csgo-datasets/              70T   64T  6.8T  91% /mnt/data/csgo-datasets
40b4a7ead0769f65bd0322230f875073e9bdb9d4b6ff190678cd1e19b73e14b9-rootfs                                   5.3T   19G  5.0T   1% /usr/lib/x86_64-linux-gnu/libcuda.so.560.35.05
tmpfs                                                                                                     931G  196K  931G   1% /proc/driver/nvidia/params
tmpfs                                                                                                     931G  4.0K  931G   1% /etc/nvidia/nvidia-application-profiles-rc.d
2.0K	data
3.5K	weights
358G	logs
108G	/mnt/nas/hj/data
241K	/mnt/workspace/hj
```

## Next Step
- Source detected env on PAI:

```bash
source "/mnt/nas/hj/H20_Video_inpainting_DPO/configs/paths/pai.detected.env"
```

- If any root above is blank or marked MISSING in the symlink section, that dataset/weight is still unconfirmed and should be downloaded or pointed to before launching the corresponding experiment.
