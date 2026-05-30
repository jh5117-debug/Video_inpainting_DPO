# PAI Audit And Asset Preparation

- generated_at: 2026-05-24T01:11:21+08:00
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

Sun May 24 01:11:22 AM CST 2026
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
?? PRD/data_and_weight_assets.md
?? PRD/data_generation_manifest_schema.md
?? PRD/dpo_diagnostics_and_metrics_plan.md
?? PRD/pai_asset_readiness_report.md
?? PRD/pai_audit_pai_node_20260523_151346.md
?? PRD/pai_audit_pai_node_20260524_001140.md
?? PRD/pai_audit_pai_node_20260524_011121.md
?? configs/
?? grit_roi_heads.py
?? official_videodpo_diffueraser_data_partialmask_loser_comp_k4/
?? official_videodpo_diffueraser_data_partialmask_loser_nocomp_k4/
?? scripts/lingbot_process.sh
?? scripts/pai_audit_and_prepare_assets.sh
?? scripts/pai_generate_fullmask_losers.sh
?? scripts/pai_generate_partialmask_losers_k4.sh
?? scripts/pai_smoke_test_generation_models.sh
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

Sun May 24 01:11:25 2026       
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
| N/A   31C    P0             76W /  700W |       0MiB / 143771MiB |      0%      Default |
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
| N/A   30C    P0             75W /  700W |       4MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA L20X                    On  |   00000000:1C:00.0 Off |                    0 |
| N/A   30C    P0             77W /  700W |     292MiB / 143771MiB |      0%      Default |
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
VIDEO_DPO_DATA_ROOT=/mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/data
YOUTUBE_VOS_ROOT=
GENERATED_LOSER_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers
DIFFUERASER_WEIGHT_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/diffueraser/Orign_Diffueraser
PROPAINTER_WEIGHT_ROOT=/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter
COCOCO_WEIGHT_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/COCOCO_weight
MINIMAX_REMOVER_WEIGHT_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO/weights/minimax_remover
OFFICIAL_VIDEODPO_WEIGHT_ROOT=/mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/checkpoints
VC2_WEIGHT_ROOT=/mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/checkpoints/vc2
```

## Prepared Current Symlinks
- LINKED: /mnt/nas/hj/H20_Video_inpainting_DPO/data/videodpo/current -> /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/data
- MISSING: /mnt/nas/hj/H20_Video_inpainting_DPO/data/youtubevos/current
- LINKED: /mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/current -> /mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers
- LINKED: /mnt/nas/hj/H20_Video_inpainting_DPO/weights/diffueraser/current -> /mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/diffueraser/Orign_Diffueraser
- LINKED: /mnt/nas/hj/H20_Video_inpainting_DPO/weights/propainter/current -> /mnt/nas/hj/data/third_party_video_inpainting/weights/propainter
- LINKED: /mnt/nas/hj/H20_Video_inpainting_DPO/weights/cococo/current -> /mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/COCOCO_weight
- LINKED: /mnt/nas/hj/H20_Video_inpainting_DPO/weights/minimax_remover/current -> /mnt/nas/hj/H20_Video_inpainting_DPO/weights/minimax_remover
- LINKED: /mnt/nas/hj/H20_Video_inpainting_DPO/weights/official_videodpo/current -> /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/checkpoints
- LINKED: /mnt/nas/hj/H20_Video_inpainting_DPO/weights/vc2/current -> /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/checkpoints/vc2

## Asset Readiness Summary
| Asset | Status | Path |
| --- | --- | --- |
| VideoDPO data | FOUND | `/mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/data` |
| YouTube-VOS data | MISSING/UNCONFIRMED | `` |
| Generated losers root | FOUND | `/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers` |
| DiffuEraser weights | FOUND | `/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/diffueraser/Orign_Diffueraser` |
| ProPainter weights | FOUND | `/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter` |
| CoCoCo weights | FOUND | `/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/COCOCO_weight` |
| MiniMax-Remover weights | FOUND | `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/minimax_remover` |
| Official VideoDPO weights | FOUND | `/mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/checkpoints` |
| VC2 weights | FOUND | `/mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/checkpoints/vc2` |

## Generated Loser Data Readiness
| Dataset | Status | Root |
| --- | --- | --- |
| official_videodpo_diffueraser_data_fullmask_loser | HAS_FILES:1 | `/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_fullmask_loser` |
| official_videodpo_diffueraser_data_partialmask_loser_k4 | HAS_FILES:1 | `/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4` |
| official_videodpo_diffueraser_youtubevos_partialmask_loser_k4 | HAS_FILES:1 | `/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4` |

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

