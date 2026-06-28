# Exp39 H20 Environment / Weight / GPU Audit

Date: 2026-06-28

Status: `H20_ENV_WEIGHT_GPU_AUDIT_COMPLETED_PARTIAL_WORKTREE_BLOCKED`

This audit used H20 read-only inspection. No H20 training was launched.

## Key Findings

- H20 host: `instance-afs92r3e`.
- GPUs: 8 x NVIDIA H20 observed idle during audit.
- `/home/nvme01`: 3.4T total, 3.1T used, 367G free, 90% used.
- H20 old repo `/home/nvme01/H20_Video_inpainting_DPO` exists and is large
  (`1.6T` observed by `du -sh`). It remains preserved.
- H20 Exp39 partial clone path was observed at 6.6M during audit, but it was not
  trusted because prior sparse checkout timed out. After the audit, the partial
  clone created by this session was removed; the H20 Exp39 worktree is currently
  absent and must be recreated cleanly before any H20 training.
- H20 `weights/minimax_remover/current` is missing.
- H20 `/home/nvme01/H20_Video_inpainting_DPO/weights` exists and is about 93G,
  but MiniMax current symlink is not present.
- Multiple Python environments report CUDA available and
  `torch.cuda.is_bf16_supported() == True` on H20.
- One env, `/home/nvme01/miniconda3/envs/twin/bin/python`, has a torch import
  error and should not be used.

## Raw Audit Excerpt

```text
host	instance-afs92r3e
date	2026-06-28T21:54:09+08:00
id	uid=1000(ubuntu) gid=0(root) groups=0(root)
gpu
0, NVIDIA H20, 28, 97871, 0
1, NVIDIA H20, 1, 97871, 0
2, NVIDIA H20, 1, 97871, 0
3, NVIDIA H20, 1, 97871, 0
4, NVIDIA H20, 1, 97871, 0
5, NVIDIA H20, 1, 97871, 0
6, NVIDIA H20, 1, 97871, 0
7, NVIDIA H20, 1, 97871, 0
compute
df
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p1  3.4T  3.1T  367G  90% /home/nvme01
/dev/sda4       341G  108G  233G  32% /home
paths
exists	/home/nvme01/H20_Video_inpainting_DPO	1.6T
drwxr-xr-x 40 ubuntu root 4096 Jun  5 04:56 /home/nvme01/H20_Video_inpainting_DPO
exists	/home/nvme01/H20_Video_inpainting_DPO_exp39_minimax_h20	6.6M
drwxr-xr-x 3 ubuntu root 4096 Jun 28 21:28 /home/nvme01/H20_Video_inpainting_DPO_exp39_minimax_h20
exists	/home/nvme01/H20_Video_inpainting_DPO/data/h20_mirror/minimax	2.6M
drwxr-xr-x 3 ubuntu root 4096 Jun 28 21:06 /home/nvme01/H20_Video_inpainting_DPO/data/h20_mirror/minimax
missing	/home/nvme01/H20_Video_inpainting_DPO/weights/minimax_remover/current
exists	/home/nvme01/H20_Video_inpainting_DPO/weights/minimax_remover	8.0K
drwxr-xr-x 2 ubuntu root 4096 May 25 10:31 /home/nvme01/H20_Video_inpainting_DPO/weights/minimax_remover
exists	/home/nvme01/H20_Video_inpainting_DPO/weights	93G
drwxr-xr-x 15 ubuntu root 4096 May 25 10:31 /home/nvme01/H20_Video_inpainting_DPO/weights
missing	/home/nvme01/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/minimax
python
PY /usr/bin/python3
TORCH 2.7.1+cu128 CUDA 12.8 AVAILABLE True
GPU0 NVIDIA H20 BF16 True CAP (9, 0)
PY /home/nvme01/miniconda3/bin/python
TORCH 2.7.1+cu126 CUDA 12.6 AVAILABLE True
GPU0 NVIDIA H20 BF16 True CAP (9, 0)
PY /home/nvme01/miniconda3/envs/SiT/bin/python
TORCH 2.7.0 CUDA None AVAILABLE False
PY /home/nvme01/miniconda3/envs/bd3lm/bin/python
TORCH 2.7.1+cu126 CUDA 12.6 AVAILABLE True
GPU0 NVIDIA H20 BF16 True CAP (9, 0)
PY /home/nvme01/miniconda3/envs/brain/bin/python
TORCH 2.5.1+cu124 CUDA 12.4 AVAILABLE True
GPU0 NVIDIA H20 BF16 True CAP (9, 0)
PY /home/nvme01/miniconda3/envs/chat/bin/python
TORCH 2.9.1+cu128 CUDA 12.8 AVAILABLE True
GPU0 NVIDIA H20 BF16 True CAP (9, 0)
PY /home/nvme01/miniconda3/envs/flux/bin/python
TORCH 2.4.1+cu121 CUDA 12.1 AVAILABLE True
GPU0 NVIDIA H20 BF16 True CAP (9, 0)
PY /home/nvme01/miniconda3/envs/hongyuqi_slam/bin/python
TORCH 2.5.1+cu124 CUDA 12.4 AVAILABLE True
GPU0 NVIDIA H20 BF16 True CAP (9, 0)
PY /home/nvme01/miniconda3/envs/hsretargeting/bin/python
TORCH 2.10.0+cu128 CUDA 12.8 AVAILABLE True
GPU0 NVIDIA H20 BF16 True CAP (9, 0)
PY /home/nvme01/miniconda3/envs/hssim/bin/python
TORCH 2.7.0+cu128 CUDA 12.8 AVAILABLE True
GPU0 NVIDIA H20 BF16 True CAP (9, 0)
PY /home/nvme01/miniconda3/envs/phystwin/bin/python
TORCH 2.10.0+cu128 CUDA 12.8 AVAILABLE True
GPU0 NVIDIA H20 BF16 True CAP (9, 0)
PY /home/nvme01/miniconda3/envs/roboos/bin/python
TORCH 2.9.1+cu128 CUDA 12.8 AVAILABLE True
GPU0 NVIDIA H20 BF16 True CAP (9, 0)
PY /home/nvme01/miniconda3/envs/scaffold_gs/bin/python
TORCH 1.12.1 CUDA 11.6 AVAILABLE True
GPU0 NVIDIA H20 BF16 True CAP (9, 0)
PY /home/nvme01/miniconda3/envs/scaffold_gs_cuda12/bin/python
TORCH 2.1.2 CUDA 12.1 AVAILABLE True
GPU0 NVIDIA H20 BF16 True CAP (9, 0)
PY /home/nvme01/miniconda3/envs/sim/bin/python
TORCH 2.7.0+cu128 CUDA 12.8 AVAILABLE True
GPU0 NVIDIA H20 BF16 True CAP (9, 0)
PY /home/nvme01/miniconda3/envs/sphere/bin/python
TORCH 2.10.0+cu128 CUDA 12.8 AVAILABLE True
GPU0 NVIDIA H20 BF16 True CAP (9, 0)
PY /home/nvme01/miniconda3/envs/t2v/bin/python
TORCH 2.5.1+cu124 CUDA 12.4 AVAILABLE True
GPU0 NVIDIA H20 BF16 True CAP (9, 0)
PY /home/nvme01/miniconda3/envs/twin/bin/python
TORCH_ERROR ImportError('/home/nvme01/miniconda3/envs/twin/lib/python3.10/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent')
PY /home/nvme01/miniconda3/envs/wan/bin/python
TORCH 2.5.1+cu124 CUDA 12.4 AVAILABLE True
GPU0 NVIDIA H20 BF16 True CAP (9, 0)
PY /home/nvme01/miniconda3/envs/wan_env/bin/python
TORCH 2.5.1+cu121 CUDA 12.1 AVAILABLE True
GPU0 NVIDIA H20 BF16 True CAP (9, 0)

```

## Decision

```text
H20_ENV_WEIGHT_GPU_AUDIT_COMPLETED_PARTIAL_WORKTREE_BLOCKED
```

H20 hardware and BF16 support look viable, but the code mirror and MiniMax
weight symlink are not ready. Do not start H20 MiniMax training until those are
fixed, the worktree is recreated cleanly, and a minimal smoke passes.
