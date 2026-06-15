# VideoPainter Adapter Gate2000 Hard Precheck

Date: Tue Jun 16 00:29:08 CST 2026
Host: dsw-753014-dc85766cb-4v2jj
Repo: /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate
Head: 2e187ee Document PAI VideoPainter gate sync block

## Final Status

status: blocked_before_trainer_preflight

sync_strategy: clean_worktree

What passed:

- Clean Exp14 worktree exists on PAI.
- Exp14 trainer and launcher static checks passed.
- VideoPainter code repo was rsynced from HAL after the initial missing-repo
  check:
  `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter`
- YouTube-VOS, DAVIS, and generated-loser manifest exist.
- Manifest does not contain `/home/nvme01`.
- GPUs are available.

Hard blocker:

```text
missing third_party/VideoPainter/ckpt/CogVideoX-5b-I2V
missing third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch
```

The trainer preflight was not run because policy/reference VideoPainter cannot
be constructed without these weights. Gate2000 was not launched.

## Narrow path checks
- MISSING: /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter
- MISSING: /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/third_party/VideoPainter
- MISSING: /mnt/workspace/hj/nas_hj/third_party/VideoPainter
- MISSING: /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp09_10_11_pai_sync/third_party/VideoPainter
- MISSING: /home/hj/dpo-2-1-exp/third_party_baselines/VideoPainter
- MISSING: /mnt/nas/hj/third_party/VideoPainter
- MISSING: /mnt/nas/hj/VideoPainter
## parent listings
### /mnt/workspace/hj/nas_hj
lrwxrwxrwx 1 root root 11 May 12 00:05 /mnt/workspace/hj/nas_hj -> /mnt/nas/hj
### /mnt/nas/hj
total 144
drwxr-xr-x 23 root   root   4096 Jun 16 00:25 .
drwxrwxrwt  9 root   root   4096 May 12 00:05 ..
drwxr-xr-x  2 root   root   4096 May 13 19:48 .hf_cache
drwxr-xr-x  5 root   root   4096 May 13 19:48 .wandb_cache
drwxr-xr-x 37 root   root   4096 Jun  5 09:49 H20_Video_inpainting_DPO
drwxrwxr-x 41   1038 1038   4096 Jun 15 04:32 H20_Video_inpainting_DPO_exp09_10_11_pai_sync
drwxr-xr-x 41 root   root   4096 Jun 16 00:26 H20_Video_inpainting_DPO_exp14_videopainter_gate
drwxrwxr-x 34 ubuntu root   4096 Jun  6 14:43 H20_Video_inpainting_DPO_exp8c_pai_sync
drwxr-xr-x 27 root   root   4096 May 15 10:19 H20_Video_inpainting_DPO_scp_backup_20260515_101902
drwxr-xr-x  4 root   root   4096 May 17 01:04 conda_envs
drwxr-xr-x  5 root   root   4096 May 14 17:31 data
drwxr-xr-x  2 root   root   4096 May 19 10:49 env_packs
-rw-r--r--  1 root   root 133126 Jun  9 23:26 exp11_impl_sync_20260609.tgz
drwxr-xr-x  3 root   root   4096 May 14 17:32 external
drwxr-xr-x  3 root   root   4096 May 16 12:13 h20_vbench_assets
drwxr-xr-x  4 root   root   4096 May 15 22:51 hf_New_DPO_data
drwxr-xr-x  3 root   root   4096 May 15 23:46 hf_fullmask_assets
drwxr-xr-x 12 root   root   4096 May 14 22:52 logs
drwxr-xr-x  4 root   root   4096 May 21 06:20 official_repos
drwxr-xr-x  5 root   root   4096 May 14 04:30 pai_pre_pull_untracked_backup_20260514_043017
drwx------  2 root   root   4096 May 12 00:05 tmp
drwx------  2 root   root   4096 May 15 11:08 transfer_logs
drwxr-xr-x 10 ubuntu root   4096 Apr 18 10:32 weights
drwxr-xr-x 18 ubuntu root   4096 May 14 06:00 world_model_phys
### /mnt/nas/hj/weights
total 5
drwxr-xr-x 10 ubuntu root 4096 Apr 18 10:32 .
drwxr-xr-x 23 root   root 4096 Jun 16 00:25 ..
drwxr-xr-x  6 ubuntu root 4096 Feb  6 10:52 PCM_Weights
drwxr-xr-x  3 root   root 4096 May 15 09:41 Qwen2.5-VL-7B-Instruct
drwxr-xr-x  3 root   root 4096 May 15 09:41 animatediff-motion-adapter-v1-5-2
drwxr-xr-x  6 ubuntu root 4096 Mar 24 15:12 diffuEraser
drwxr-xr-x  2 root   root 4096 May 15 09:41 metrics
drwxr-xr-x  2 root   root 4096 Jun  7 23:37 propainter
drwxr-xr-x  3 ubuntu root 4096 Feb  6 10:47 sd-vae-ft-mse
drwxr-xr-x 10 ubuntu root 4096 Feb  6 10:46 stable-diffusion-v1-5
### /mnt/nas/hj/data/third_party_video_inpainting
total 2
drwxr-xr-x 3 root root 4096 May 14 17:31 .
drwxr-xr-x 5 root root 4096 May 14 17:31 ..
lrwxrwxrwx 1 root root   71 Apr 29 09:15 envs -> /home/nvme01/H20_Video_inpainting_DPO/third_party_video_inpainting/envs
lrwxrwxrwx 1 root root   72 Apr 29 09:15 repos -> /home/nvme01/H20_Video_inpainting_DPO/third_party_video_inpainting/repos
drwxr-xr-x 4 root root 4096 May 14 17:31 weights
### /mnt/workspace/hj/nas_hj/weights
total 5
drwxr-xr-x 10 ubuntu root 4096 Apr 18 10:32 .
drwxr-xr-x 23 root   root 4096 Jun 16 00:25 ..
drwxr-xr-x  6 ubuntu root 4096 Feb  6 10:52 PCM_Weights
drwxr-xr-x  3 root   root 4096 May 15 09:41 Qwen2.5-VL-7B-Instruct
drwxr-xr-x  3 root   root 4096 May 15 09:41 animatediff-motion-adapter-v1-5-2
drwxr-xr-x  6 ubuntu root 4096 Mar 24 15:12 diffuEraser
drwxr-xr-x  2 root   root 4096 May 15 09:41 metrics
drwxr-xr-x  2 root   root 4096 Jun  7 23:37 propainter
drwxr-xr-x  3 ubuntu root 4096 Feb  6 10:47 sd-vae-ft-mse
drwxr-xr-x 10 ubuntu root 4096 Feb  6 10:46 stable-diffusion-v1-5
## data
- OK: /mnt/workspace/hj/nas_hj/data/external/youtubevos_432_240
- OK: /mnt/workspace/hj/nas_hj/data/external/youtubevos_432_240_eval100
- OK: /mnt/workspace/hj/nas_hj/data/external/davis_432_240
- OK: /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4/manifests/selected_primary_comp.repaired.pai_paths.jsonl
- MISSING: /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4/manifests/selected_primary_comp.repaired.pai_paths.jsonl
## GPU
0, 0, 143771, 0
1, 0, 143771, 0
2, 0, 143771, 0
3, 0, 143771, 0
4, 244, 143771, 0
5, 4, 143771, 0
6, 292, 143771, 0
7, 58071, 143771, 0

## VideoPainter weight path checks
- MISSING: /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter/ckpt/CogVideoX-5b-I2V
- MISSING: /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch
- MISSING: /mnt/nas/hj/weights/CogVideoX-5b-I2V
- MISSING: /mnt/nas/hj/weights/VideoPainter
- MISSING: /mnt/nas/hj/weights/videopainter
- MISSING: /mnt/nas/hj/weights/cogvideox
- MISSING: /mnt/nas/hj/official_repos/VideoPainter
- MISSING: /mnt/nas/hj/data/third_party_video_inpainting/weights/VideoPainter
- MISSING: /mnt/nas/hj/data/third_party_video_inpainting/weights/videopainter
- MISSING: /mnt/nas/hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter/ckpt/CogVideoX-5b-I2V
## nearby official_repos
- /mnt/nas/hj/official_repos
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/.git
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/.git/branches
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/.git/hooks
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/.git/info
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/.git/logs
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/.git/objects
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/.git/refs
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/OmniScore
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/assets
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/assets/vc2-dpo
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/assets/vc2-init
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/checkpoints
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/checkpoints/vc2
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/configs
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/configs/inference
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/configs/t2v_turbo_dpo
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/configs/vc2_dpo
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/data
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/data/__pycache__
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/log_image
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/log_image/images
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/lvdm
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/lvdm/__pycache__
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/lvdm/models
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/lvdm/modules
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/lvdm/samplers
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/prompts
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/scripts
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/scripts/__pycache__
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/scripts/turbo_inference
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/scripts_sh
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/utils
- /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/utils/__pycache__
- /mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4
- /mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4/.git
- /mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4/.git/branches
- /mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4/.git/hooks
- /mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4/.git/info
- /mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4/.git/logs
- /mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4/.git/objects
- /mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4/.git/refs
- /mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4/OmniScore
- /mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4/assets
- /mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4/assets/vc2-dpo
- /mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4/assets/vc2-init
- /mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4/configs
- /mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4/configs/inference
- /mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4/configs/t2v_turbo_dpo
- /mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4/configs/vc2_dpo
- /mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4/data
- /mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4/data/__pycache__
- /mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4/lvdm
- /mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4/lvdm/models
- /mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4/lvdm/modules
- /mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4/lvdm/samplers
- /mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4/prompts
- /mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4/scripts
- /mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4/scripts/__pycache__
- /mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4/scripts/turbo_inference
- /mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4/scripts_sh
- /mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4/utils
- /mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4/utils/__pycache__
## selected ckpt-like dirs under weights
- /mnt/nas/hj/weights/propainter
- /mnt/nas/hj/weights/stable-diffusion-v1-5
- /mnt/nas/hj/weights/stable-diffusion-v1-5/.cache
- /mnt/nas/hj/weights/stable-diffusion-v1-5/.cache/huggingface
- /mnt/nas/hj/weights/stable-diffusion-v1-5/feature_extractor
- /mnt/nas/hj/weights/stable-diffusion-v1-5/safety_checker
- /mnt/nas/hj/weights/stable-diffusion-v1-5/scheduler
- /mnt/nas/hj/weights/stable-diffusion-v1-5/text_encoder
- /mnt/nas/hj/weights/stable-diffusion-v1-5/tokenizer
- /mnt/nas/hj/weights/stable-diffusion-v1-5/unet
- /mnt/nas/hj/weights/stable-diffusion-v1-5/vae
