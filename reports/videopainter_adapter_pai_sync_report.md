# VideoPainter Adapter PAI Sync Report

Date: Tue Jun 16 00:26:07 CST 2026
Host: dsw-753014-dc85766cb-4v2jj

sync_strategy: clean_worktree
source_commit: 2e187ee
target_repo: /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate
priority_dirty_repo: /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO
exp14_files_current: yes

This clean directory was used because the priority PAI repo contains tracked and
untracked local files that block ff-only pull. No reset, clean, deletion, or
overwrite was performed on the dirty priority repo.

VideoPainter code was not present on PAI, and the PAI GitHub clone attempt
failed with a TLS recv error. The code repo was then rsynced from HAL:

```text
source = /home/hj/dpo-2-1-exp/third_party_baselines/VideoPainter
target = /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter
```

This synced code only. It did not sync checkpoints or generated outputs.

Current hard blocker:

```text
missing = third_party/VideoPainter/ckpt/CogVideoX-5b-I2V
missing = third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch
```

The trainer preflight and gate2000 were not launched.
