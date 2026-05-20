# PAI/H20 VideoDPO Migration And Runbook

Date: 2026-05-15

This document continues the previous VideoDPO/VC2 reproduction, DiffuEraser full-mask bridge, H20-to-PAI migration, health-check, and Hugging Face asset-transfer notes.  It is written as a practical runbook for the next session.

Read together with:

- `PRD/CURRENT_STATUS_20260512.md`
- `PRD/meeting_followup_videodpo_repro_and_bridge_20260511.md`
- `PRD/DPO_Training_Metrics_Explained.md`

## 1. 信息需要向管理员确认什么

管理员已经给出的资源说明：

- `cpu-machine`: CPU 按量实例，用于数据预处理、代码测试等，不用可以停机。
- `lingbot_fast_multi`: GPU 机型，8x H200，用于训练。
- 只读数据集：
  - `/mnt/data/csgo-datasets/`: CPFS 高速盘，脏数据大部分已剔除。
  - `/mnt/data/csgo-datasets-oss/`: OSS 访问，内容基本一致但脏数据未完全删除。
- 持久化硬盘：
  - `/mnt/nas/`: NAS，放持久化 checkpoint、代码、数据处理中间产物。
  - `/mnt/data/pku/`: CPFS，加速训练，只放启动训练必须内容和最近 ckpt，用完同步回 NAS。

需要再问管理员的关键信息如下。

### 1.1 机器和 GPU

- 当前登录的 PAI 机器到底是不是 `lingbot_fast_multi`。
  - 我们在 PAI health check 里看到的是 8x `NVIDIA L20X`。
  - 管理手册写的是 8x H200。
  - 需要确认是机器拿错了、显示名不同，还是资源池不同。
- 8 卡训练是否需要平台调度命令，还是直接在 shell 里用 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`。
- 有没有最大运行时长、抢占规则、空闲回收规则。
- 是否推荐用 `tmux`、`nohup`、平台 job 管理器，还是必须通过某个提交入口启动长任务。
- 重启后 GPU 机器系统盘重置，是否 `/mnt/nas` 和 `/mnt/data/pku` 一定保留。
- 是否可以申请额外按量 GPU 机器做数据预处理或 VBench 评测。

### 1.2 存储和路径

- 给个人项目的标准目录是什么。
  - 建议确认是否统一使用：
    - NAS: `/mnt/nas/hj`
    - CPFS 快盘: `/mnt/data/pku/hj`
- `/mnt/workspace/hj/nas_hj` 是否只是 `/mnt/nas/hj` 的软链接或 bind mount。
  - 之前实际检查到：
    - `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO`
    - `readlink -f` 后指向 `/mnt/nas/hj/H20_Video_inpainting_DPO`
- NAS、CPFS 的容量 quota 和 inode quota。
- CPFS `/mnt/data/pku/` 的收费/清理策略。
- checkpoint 应该多久同步回 NAS。
- CPFS 里是否可以放完整权重，还是只放训练当前需要的权重和最近 ckpt。
- 多人共享盘是否有推荐权限模式，避免 root 写入后普通用户不可写。

### 1.3 网络、代理和 Hugging Face

- PAI 是否可以直接访问 `huggingface.co`，还是必须用 `HF_ENDPOINT=https://hf-mirror.com`。
- `hf-mirror.com` 对刚上传的新文件是否会延迟同步。
  - 本次 PAI 用 mirror 列 repo 文件时返回 0 个新分片，很可能是 mirror lag。
  - 需要知道什么时候该 `unset HF_ENDPOINT` 用官方站点。
- 是否允许长时间大文件下载。
- 是否支持 `hf_xet`，是否有推荐版本。
- 是否有集群内代理地址，例如 `http_proxy`、`https_proxy`、`all_proxy`。
- Hugging Face token 应该放哪里，是否可以在共享机器里 `hf auth login`。
- 是否有 GitHub 访问限制。
  - SSH clone 在 PAI 上失败过：`Permission denied (publickey)`。
  - HTTPS clone 可以用。

### 1.4 环境和系统包

- Conda 环境应该建在哪里。
  - 建议 `/mnt/nas/hj/conda_envs/diffueraser`，因为系统盘重启会重置。
- 是否已有平台推荐 CUDA/PyTorch 镜像。
- 是否允许 `apt install` 或只能 `pip/conda`。
- 是否已有 `zstd`、`rsync`、`git-lfs`、`ripgrep`。
  - PAI health check 里 `rg` 缺失，导致 conflict-marker scan 被跳过。
- 是否可以保存系统镜像，镜像里应该包含哪些系统包。

### 1.5 数据合规

- `/mnt/data/csgo-datasets` 和 `/mnt/data/csgo-datasets-oss` 明确只读，不能改原始数据。
- 派生数据、清洗后索引、cache、metadata 应该放 `/mnt/nas/hj` 还是 `/mnt/data/pku/hj`。
- 是否可以把只读数据集中的样本复制到个人目录做 smoke。

## 2. 当前任务线

目前有两条线。

### 2.1 Task 1: 官方 VideoDPO/VC2 复现

目标：

1. 复现官方 VideoDPO 的 VC2-DPO 训练。
2. 用 VBench 评测，形成可对齐论文的复现实验。

H20 上已经完成一次等价 epoch-10 的训练。

关键结论：

- 原始 run 在大约 epoch 3 后因为 OOM/中断留下了坏的 `last.ckpt`。
- 有效中间 ckpt 是 `step-step=2495.ckpt`。
- 之后用 continuation run 从该权重继续。
- continuation run 成功停止于 `max_steps=3755 reached`。
- 总等价 official step 约为 `2495 + 3755 = 6250`，对应 10 个 epoch 的训练要求。

### 2.2 Task 2: VideoDPO 到 DiffuEraser full-mask bridge

目标：

1. 保持 VideoDPO 数据和 task 不变。
2. 把模型替换成 DiffuEraser full-mask 形式。
3. 用最小变量验证桥接训练是否能稳定跑通。

目前 fullmask bridge 必须对齐 official VideoDPO 设置：

- `320x512`
- `video_length=16`
- `frame_stride=1`
- `batch=1`
- `grad_accum=2`
- `num_workers=16`
- `lr=6e-6`
- `beta=5000`
- `precision=no`，即不使用 bf16 mixed precision
- `ckpt_every=499`
- `split_pos_neg=0`

不要再手动保留这些旧设置：

- `SPLIT_POS_NEG_FORWARD=1`
- `MIXED_PRECISION=bf16`
- `RESOLUTION=512`

这三个设置会让 fullmask bridge 偏离 official VideoDPO setting，其中 `SPLIT_POS_NEG_FORWARD=1` 在 H20 上已知容易 step-0 `SIGFPE`。

## 3. H20 官方 VideoDPO 复现状态

### 3.1 原始 run

```text
RUN_NAME=h20-vc2-dpo-official-full-gpu0-7_20260514_035204
RUN_DIR=/home/nvme01/H20_Video_inpainting_DPO/logs/videodpo_vc2_dpo/h20-vc2-dpo-official-full-gpu0-7_20260514_035204
```

失败原因：

- `last.ckpt` 损坏。
- `torch.load(last.ckpt)` 报：

```text
RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory
```

有效的 trainstep checkpoints：

```text
step-step=0499.ckpt   epoch=0 global_step=499
step-step=0998.ckpt   epoch=1 global_step=998
step-step=1497.ckpt   epoch=2 global_step=1497
step-step=1996.ckpt   epoch=3 global_step=1996
step-step=2495.ckpt   epoch=3 global_step=2495
```

最新有效 ckpt：

```text
/home/nvme01/H20_Video_inpainting_DPO/logs/videodpo_vc2_dpo/h20-vc2-dpo-official-full-gpu0-7_20260514_035204/checkpoints/trainstep_checkpoints/step-step=2495.ckpt
```

### 3.2 continuation run

```text
RUN_NAME=h20-vc2-dpo-official-full-gpu0-7_resume-step2495_20260514_195149
RUN_DIR=/home/nvme01/H20_Video_inpainting_DPO/logs/videodpo_vc2_dpo/h20-vc2-dpo-official-full-gpu0-7_resume-step2495_20260514_195149
```

启动时使用：

```bash
MAX_OPT_STEPS=3755
CONFIG="$RESUME_CONFIG"
```

注意解释：

- 终端进度条显示 `Epoch 0/1/2/...` 是 continuation run 自己的新 trainer epoch 计数。
- 不是从原始模型重新训练。
- 因为 continuation run 是新目录、新 trainer，它会从 `Epoch 0` 开始显示。
- 训练内容从 `step-step=2495.ckpt` 的模型权重开始接着优化。
- 最终停止条件是 `Trainer.fit stopped: max_steps=3755 reached`。
- 这等价于原始训练已经跑了 2495 步，再补 3755 步，总共约 6250 步。

终端完成标志：

```text
`Trainer.fit` stopped: `max_steps=3755` reached.
[vc2-train] done
```

### 3.3 H20 checkpoint audit

已审查的 ckpt：

```text
final_epoch10_equiv:
  path: /home/nvme01/H20_Video_inpainting_DPO/logs/videodpo_vc2_dpo/h20-vc2-dpo-official-full-gpu0-7_resume-step2495_20260514_195149/checkpoints/last.ckpt
  size: 17.427 GB
  torch.load: OK
  epoch: 6
  global_step: 3755
  state_dict_keys: 2039

base_model:
  path: /home/nvme01/H20_Video_inpainting_DPO/external/VideoDPO/checkpoints/vc2/model.ckpt
  size: 6.896 GB
  torch.load: OK
  state_dict_keys: 2039

ref_model:
  path: /home/nvme01/H20_Video_inpainting_DPO/external/VideoDPO/checkpoints/vc2/ref_model.ckpt
  size: 5.265 GB
  torch.load: OK
  state_dict_keys: 1484
```

结论：

- H20 final ckpt 可加载。
- base/ref ckpt 可加载。
- VideoDPO 训练本身已经达到原始要求的结束条件。
- 还缺 VBench 生成和评测。

## 4. PAI 代码迁移和 Git

### 4.1 不建议直接把 scp 目录变成 git repo

之前 PAI 是一天前从 H20 整体 scp 过去的目录，可能没有 `.git` 或者内容混乱。推荐做法是：

1. 把旧目录改名为 backup。
2. 用 GitHub 重新 clone 干净代码。
3. 只把大数据、权重、日志从 backup 或 NAS 大目录迁回。

SSH clone 在 PAI 上失败过：

```text
git@github.com: Permission denied (publickey).
```

因此先用 HTTPS：

```bash
cd /mnt/nas/hj

mv H20_Video_inpainting_DPO H20_Video_inpainting_DPO_scp_backup_$(date +%Y%m%d_%H%M%S)

git clone --recursive https://github.com/jh5117-debug/Video_inpainting_DPO.git H20_Video_inpainting_DPO
cd H20_Video_inpainting_DPO
git rev-parse --short HEAD
```

如果 repo 已经存在：

```bash
cd /mnt/nas/hj/H20_Video_inpainting_DPO
git remote set-url origin https://github.com/jh5117-debug/Video_inpainting_DPO.git
git pull --ff-only origin main
git submodule update --init --recursive
git rev-parse --short HEAD
```

### 4.2 当前 PAI 代码状态

PAI 已经拉到至少：

```text
b57cef1 Treat VC2 checkpoints as optional for fullmask health
```

关键脚本：

```text
DPO_finetune/scripts/pai_videodpo_fullmask_health_check.sh
DPO_finetune/scripts/sc_videodpo_fullmask_diffueraser_stage1.sbatch
external/VideoDPO
external/VBench
```

注意：

- `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO` 实际解析到 `/mnt/nas/hj/H20_Video_inpainting_DPO`。
- 文档和命令建议统一写 `/mnt/nas/hj/H20_Video_inpainting_DPO`，减少混乱。

## 5. PAI 健康检查

### 5.1 基础命令

```bash
cd /mnt/nas/hj/H20_Video_inpainting_DPO
git pull --ff-only origin main
git rev-parse --short HEAD
```

运行 health check：

```bash
WEIGHTS_DIR=/mnt/nas/hj/weights \
DPO_DATA_ROOT=/mnt/nas/hj/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml \
bash DPO_finetune/scripts/pai_videodpo_fullmask_health_check.sh 2>&1 | tee /tmp/pai_health_check.log

grep -E '^\[FAIL\]|^\[WARN\]' /tmp/pai_health_check.log
```

如果环境已经建好：

```bash
DIFFUERASER_ENV=/mnt/nas/hj/conda_envs/diffueraser \
WEIGHTS_DIR=/mnt/nas/hj/weights \
DPO_DATA_ROOT=/mnt/nas/hj/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml \
bash DPO_finetune/scripts/pai_videodpo_fullmask_health_check.sh 2>&1 | tee /tmp/pai_health_check.log
```

### 5.2 health check 检查什么

脚本会检查：

- git repo 和 working tree。
- submodules 和 code layout。
- bash syntax。
- Python key files syntax。
- fullmask bridge 默认设置：
  - `TRAIN_HEIGHT=320`
  - `TRAIN_WIDTH=512`
  - `MIXED_PRECISION=no`
  - `SPLIT_POS_NEG_FORWARD=0`
  - `BETA_DPO=5000`
  - `LR=6e-6`
  - `CKPT_STEPS=499`
  - `NUM_WORKERS=16`
- GPU 可见性。
- DiffuEraser env。
- VideoDPO env，只有 official VideoDPO/VBench 必需。
- fullmask 必需权重文件。
- VideoDPO VC2 data yaml 和 dataset root。
- fullmask dataset smoke。

VC2 base/ref checkpoint 对 fullmask 只是 warning：

```text
[WARN] VC2 base model.ckpt (official VideoDPO only) missing
[WARN] VC2 ref_model.ckpt (official VideoDPO only) missing
```

如果只跑 fullmask DiffuEraser bridge，这两个不是 blocker。若要在 PAI 跑 official VideoDPO 或 VBench，则需要补齐。

### 5.3 当前 PAI health check 观察

已观察到：

- Git repo OK。
- submodule OK。
- fullmask setting guard OK。
- GPU 检测到 8 张卡，但型号是 `NVIDIA L20X`，需要和管理员确认。
- `rg` 未安装，只影响 conflict-marker scan。
- DiffuEraser env 不存在。
- `/mnt/nas/hj/weights` 下目录存在，但部分关键文件最初为空或缺失。
- PAI 上 VideoDPO data yaml 存在于：

```text
/mnt/nas/hj/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml
```

如果 health check 仍无法 resolve data yaml，先检查这个 yaml 里是否还写着 H20 的绝对路径。

## 6. PAI 数据和权重位置

### 6.1 推荐目录

```text
Code:
  /mnt/nas/hj/H20_Video_inpainting_DPO

Weights:
  /mnt/nas/hj/weights

VideoDPO data yaml:
  /mnt/nas/hj/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml

HF download cache:
  /mnt/nas/hj/H20_Video_inpainting_DPO/.hf_cache

HF downloaded assets:
  /mnt/nas/hj/hf_New_DPO_data

Expanded fullmask assets:
  /mnt/nas/hj/hf_fullmask_assets

Conda env:
  /mnt/nas/hj/conda_envs/diffueraser
```

### 6.2 权重目录需要长这样

fullmask DiffuEraser bridge 需要：

```text
/mnt/nas/hj/weights/stable-diffusion-v1-5/model_index.json
/mnt/nas/hj/weights/stable-diffusion-v1-5/tokenizer/vocab.json
/mnt/nas/hj/weights/stable-diffusion-v1-5/tokenizer/merges.txt
/mnt/nas/hj/weights/stable-diffusion-v1-5/tokenizer/tokenizer_config.json
/mnt/nas/hj/weights/stable-diffusion-v1-5/scheduler/scheduler_config.json
/mnt/nas/hj/weights/stable-diffusion-v1-5/text_encoder/config.json
/mnt/nas/hj/weights/stable-diffusion-v1-5/text_encoder/model.safetensors or pytorch_model.bin
/mnt/nas/hj/weights/stable-diffusion-v1-5/unet/config.json
/mnt/nas/hj/weights/stable-diffusion-v1-5/unet/diffusion_pytorch_model.safetensors or .bin
/mnt/nas/hj/weights/sd-vae-ft-mse/config.json
/mnt/nas/hj/weights/sd-vae-ft-mse/diffusion_pytorch_model.safetensors or .bin
/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000/unet_main/config.json
/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000/unet_main/diffusion_pytorch_model.safetensors or .bin
/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000/brushnet/config.json
/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000/brushnet/diffusion_pytorch_model.safetensors or .bin
```

检查命令：

```bash
find /mnt/nas/hj/weights/stable-diffusion-v1-5 -maxdepth 3 -type f | head -50
find /mnt/nas/hj/weights/sd-vae-ft-mse -maxdepth 2 -type f | head -20
find /mnt/nas/hj/weights/diffuEraser/converted_weights_step48000 -maxdepth 3 -type f | head -30
```

## 7. Hugging Face 资产传输

### 7.1 原始 HF dataset 的情况

HF repo:

```text
JiaHuang01/New_DPO_data
```

网页上显示主要文件：

```text
DPO_Finetune_data.tar.gz  129 GB
```

因此下面这种 `allow_patterns` 不会拿到任何文件：

```python
allow_patterns=[
    "weights/stable-diffusion-v1-5/**",
    "weights/sd-vae-ft-mse/**",
    "weights/diffuEraser/converted_weights_step48000/**",
]
```

原因是 HF repo 里不是展开后的 `weights/...` 目录，而是一个 tar 包。

### 7.2 H20 已经打包 fullmask 必需资产

H20 上创建了最小 fullmask 资产包，不再传整个 129G 或 150G 大包。

路径：

```text
/home/nvme01/H20_Video_inpainting_DPO/data/hf_upload/fullmask_required_assets_20260515_145007.tar.zst
```

大小：

```text
25G
```

包内包含：

- fullmask 必需权重。
- DiffuEraser conda environment export。
- DiffuEraser pip freeze。
- `MANIFEST.txt`。

打包命令记录：

```bash
cd /home/nvme01/H20_Video_inpainting_DPO

ASSET_ROOT=/home/nvme01/H20_Video_inpainting_DPO/data/hf_upload/fullmask_required_assets
rm -rf "$ASSET_ROOT"
mkdir -p "$ASSET_ROOT"

rsync -a --relative \
  weights/stable-diffusion-v1-5/model_index.json \
  weights/stable-diffusion-v1-5/tokenizer \
  weights/stable-diffusion-v1-5/scheduler \
  weights/stable-diffusion-v1-5/text_encoder \
  weights/stable-diffusion-v1-5/unet \
  weights/sd-vae-ft-mse \
  weights/diffuEraser/converted_weights_step48000 \
  "$ASSET_ROOT"/

mkdir -p "$ASSET_ROOT/env"
conda env export -p /home/nvme01/conda_envs/diffueraser --no-builds \
  | grep -v '^prefix:' > "$ASSET_ROOT/env/diffueraser_environment.yml"
/home/nvme01/conda_envs/diffueraser/bin/python -m pip freeze \
  > "$ASSET_ROOT/env/diffueraser_pip_freeze.txt"

find "$ASSET_ROOT" -type f -printf '%s %P\n' | sort > "$ASSET_ROOT/MANIFEST.txt"

cd /home/nvme01/H20_Video_inpainting_DPO/data/hf_upload
tar -I 'zstd -T0 -3' -cf fullmask_required_assets_$(date +%Y%m%d_%H%M%S).tar.zst fullmask_required_assets
```

### 7.3 单个 25G 文件上传卡 99% 的原因

单个 `.tar.zst` 上传到 HF 时卡在 99%，可能卡在 Xet finalize/commit 阶段，不一定是数据没有传完。

150G Lingbot 权重上传快，是因为它是多个 safetensors shard。HF/Xet 可以并行处理多个较大但独立的对象。

单个 25G tar 是一个大对象：

- 并行度低。
- finalize 失败后很难局部重试。
- 网络代理稍微不稳定就会长时间卡住。

解决方式：分片上传。

### 7.4 H20 分片上传已经成功

分片目录：

```text
fullmask_required_assets_parts_20260515_145007
```

上传成功 commit：

```text
https://huggingface.co/datasets/JiaHuang01/New_DPO_data/commit/2680dace5b4dc6f56e9999a6c5330f05ef50eaab
```

上传日志显示：

```text
Processing Files (8 / 8): 100%
New Data Upload: 100%
✓ Uploaded
```

分片上传命令记录：

```bash
cd /home/nvme01/H20_Video_inpainting_DPO/data/hf_upload

FILE=fullmask_required_assets_20260515_145007.tar.zst
PART_DIR=fullmask_required_assets_parts_20260515_145007
rm -rf "$PART_DIR"
mkdir -p "$PART_DIR"

split -b 4G -d -a 3 "$FILE" "$PART_DIR/${FILE}.part-"
sha256sum "$FILE" > "$PART_DIR/${FILE}.sha256"

source /home/nvme01/clash-for-linux/clash.sh && proxy_on
export HF_HOME=/home/nvme04/hf_cache
unset HF_ENDPOINT

HF=/home/nvme03/workspace/world_model_phys/.conda_envs/phys-main/bin/hf

LOG=/home/nvme04/hf_upload_fullmask_assets_parts_$(date +%F_%H%M%S).log
nohup "$HF" upload JiaHuang01/New_DPO_data \
  "$PART_DIR" \
  "$PART_DIR" \
  --type dataset \
  --commit-message "Upload fullmask required assets parts" \
  > "$LOG" 2>&1 &

echo $! > /home/nvme04/hf_upload_fullmask_assets_parts.pid
tail -f "$LOG"
```

If retry is needed, rerun the same upload command. HF/Xet should reuse already uploaded chunks.

### 7.5 PAI 下载分片

如果使用 `HF_ENDPOINT=https://hf-mirror.com` 返回 0 个文件，优先怀疑 mirror 未同步新 commit。先用官方 HF 列文件。

```bash
cd /mnt/nas/hj/H20_Video_inpainting_DPO

python3 -m pip install -U huggingface_hub hf_xet

unset HF_ENDPOINT
export HF_HOME=/mnt/nas/hj/H20_Video_inpainting_DPO/.hf_cache
export HF_HUB_CACHE=/mnt/nas/hj/H20_Video_inpainting_DPO/.hf_cache

python3 - <<'PY'
from huggingface_hub import HfApi

repo_id = "JiaHuang01/New_DPO_data"
files = HfApi().list_repo_files(repo_id, repo_type="dataset")
for f in files:
    if "fullmask_required_assets" in f or "part-" in f or f.endswith(".tar.zst"):
        print(f)
PY
```

确认能看到：

```text
fullmask_required_assets_parts_20260515_145007/fullmask_required_assets_20260515_145007.tar.zst.part-000
...
fullmask_required_assets_parts_20260515_145007/fullmask_required_assets_20260515_145007.tar.zst.part-006
fullmask_required_assets_parts_20260515_145007/fullmask_required_assets_20260515_145007.tar.zst.sha256
```

再下载：

```bash
cd /mnt/nas/hj/H20_Video_inpainting_DPO

unset HF_ENDPOINT
export HF_HOME=/mnt/nas/hj/H20_Video_inpainting_DPO/.hf_cache
export HF_HUB_CACHE=/mnt/nas/hj/H20_Video_inpainting_DPO/.hf_cache

python3 - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="JiaHuang01/New_DPO_data",
    repo_type="dataset",
    allow_patterns=["fullmask_required_assets_parts_20260515_145007/**"],
    local_dir="/mnt/nas/hj/hf_New_DPO_data",
)
PY
```

如果官方 HF 太慢，再切 mirror：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

但前提是 mirror 已经能 list 到分片。

### 7.6 PAI 合并、校验、解压、同步权重

```bash
cd /mnt/nas/hj/hf_New_DPO_data/fullmask_required_assets_parts_20260515_145007

cat fullmask_required_assets_20260515_145007.tar.zst.part-* \
  > /mnt/nas/hj/hf_New_DPO_data/fullmask_required_assets_20260515_145007.tar.zst

cd /mnt/nas/hj/hf_New_DPO_data
sha256sum -c fullmask_required_assets_parts_20260515_145007/fullmask_required_assets_20260515_145007.tar.zst.sha256

mkdir -p /mnt/nas/hj/hf_fullmask_assets
tar -I zstd -xf fullmask_required_assets_20260515_145007.tar.zst \
  -C /mnt/nas/hj/hf_fullmask_assets

mkdir -p /mnt/nas/hj/weights
rsync -ah --ignore-existing \
  /mnt/nas/hj/hf_fullmask_assets/fullmask_required_assets/weights/ \
  /mnt/nas/hj/weights/
```

然后重跑 health check：

```bash
cd /mnt/nas/hj/H20_Video_inpainting_DPO

WEIGHTS_DIR=/mnt/nas/hj/weights \
DPO_DATA_ROOT=/mnt/nas/hj/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml \
bash DPO_finetune/scripts/pai_videodpo_fullmask_health_check.sh 2>&1 | tee /tmp/pai_health_check_after_assets.log

grep -E '^\[FAIL\]|^\[WARN\]' /tmp/pai_health_check_after_assets.log
```

## 8. PAI DiffuEraser 环境

fullmask 资产包里有 H20 diffueraser 环境记录：

```text
/mnt/nas/hj/hf_fullmask_assets/fullmask_required_assets/env/diffueraser_environment.yml
/mnt/nas/hj/hf_fullmask_assets/fullmask_required_assets/env/diffueraser_pip_freeze.txt
```

建议在持久化 NAS 里创建环境：

```bash
conda env create \
  -p /mnt/nas/hj/conda_envs/diffueraser \
  -f /mnt/nas/hj/hf_fullmask_assets/fullmask_required_assets/env/diffueraser_environment.yml
```

如果环境解不出来，再按 `diffueraser_pip_freeze.txt` 手动补包。

创建后检查：

```bash
/mnt/nas/hj/conda_envs/diffueraser/bin/python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
PY
```

带环境重跑 health check：

```bash
cd /mnt/nas/hj/H20_Video_inpainting_DPO

DIFFUERASER_ENV=/mnt/nas/hj/conda_envs/diffueraser \
WEIGHTS_DIR=/mnt/nas/hj/weights \
DPO_DATA_ROOT=/mnt/nas/hj/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml \
bash DPO_finetune/scripts/pai_videodpo_fullmask_health_check.sh 2>&1 | tee /tmp/pai_health_check_env.log
```

## 9. PAI fullmask training smoke

health check 通过后，先跑 1 卡 2 step smoke。

```bash
cd /mnt/nas/hj/H20_Video_inpainting_DPO

VC2_YAML=/mnt/nas/hj/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml

CUDA_VISIBLE_DEVICES=7 \
DIFFUERASER_ENV=/mnt/nas/hj/conda_envs/diffueraser \
CONDA_ENV=/mnt/nas/hj/conda_envs/diffueraser \
PROJECT_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO \
PROJECT_HOME=/mnt/nas/hj \
PROJECT_DEV=/mnt/nas/hj \
PROJECT_DATA=/mnt/nas/hj/H20_Video_inpainting_DPO \
DATA=/mnt/nas/hj/H20_Video_inpainting_DPO \
WEIGHTS_DIR=/mnt/nas/hj/weights \
DPO_DATA_ROOT="$VC2_YAML" \
RUN_NAME=pai-gpu7-videodpo-fullmask-diffueraser-smoke-$(date +%Y%m%d_%H%M%S) \
NUM_GPUS=1 \
MAX_STEPS=2 \
CKPT_STEPS=999999 \
VAL_STEPS=999999 \
LOGGING_STEPS=1 \
REPORT_TO=none \
bash DPO_finetune/scripts/sc_videodpo_fullmask_diffueraser_stage1.sbatch
```

启动日志必须看到类似：

```text
precision=no train_size=320x512 split_pos_neg=0
```

如果看到：

```text
precision=bf16
split_pos_neg=1
train_size=512x512
```

就说明又混入了旧参数，需要停掉重来。

## 10. PAI official VideoDPO/VBench 资产

fullmask smoke 不需要 VC2 official base/ref ckpt。official VideoDPO/VBench 需要：

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/external/VideoDPO/checkpoints/vc2/model.ckpt
/mnt/nas/hj/H20_Video_inpainting_DPO/external/VideoDPO/checkpoints/vc2/ref_model.ckpt
```

以及 H20 final ckpt：

```text
logs/videodpo_vc2_dpo/h20-vc2-dpo-official-full-gpu0-7_resume-step2495_20260514_195149/checkpoints/last.ckpt
```

建议从 H20 另打一个 VBench 专用包传 HF。不要把整个 1T logs 传过去。

### 10.1 H20 打包 VideoDPO epoch-10/VBench 资产

```bash
cd /home/nvme01/H20_Video_inpainting_DPO

PY=/home/nvme01/conda_envs/videodpo/bin/python

ORIG_RUN=h20-vc2-dpo-official-full-gpu0-7_20260514_035204
RESUME_RUN=h20-vc2-dpo-official-full-gpu0-7_resume-step2495_20260514_195149

ORIG_DIR=/home/nvme01/H20_Video_inpainting_DPO/logs/videodpo_vc2_dpo/$ORIG_RUN
RESUME_DIR=/home/nvme01/H20_Video_inpainting_DPO/logs/videodpo_vc2_dpo/$RESUME_RUN

FINAL_CKPT=$RESUME_DIR/checkpoints/last.ckpt
BASE_CKPT=/home/nvme01/H20_Video_inpainting_DPO/external/VideoDPO/checkpoints/vc2/model.ckpt
REF_CKPT=/home/nvme01/H20_Video_inpainting_DPO/external/VideoDPO/checkpoints/vc2/ref_model.ckpt

ASSET_ROOT=/home/nvme01/H20_Video_inpainting_DPO/data/hf_upload/videodpo_vc2_epoch10_vbench_assets
rm -rf "$ASSET_ROOT"
mkdir -p "$ASSET_ROOT/checkpoints/vc2" "$ASSET_ROOT/logs" "$ASSET_ROOT/env" "$ASSET_ROOT/configs"

cp -a "$FINAL_CKPT" "$ASSET_ROOT/checkpoints/vc2/final_epoch10_equiv_last.ckpt"
cp -a "$BASE_CKPT" "$ASSET_ROOT/checkpoints/vc2/model.ckpt"
cp -a "$REF_CKPT" "$ASSET_ROOT/checkpoints/vc2/ref_model.ckpt"

cp -a "$RESUME_DIR/configs" "$ASSET_ROOT/logs/resume_configs"
cp -a "$ORIG_DIR/configs" "$ASSET_ROOT/logs/orig_configs" 2>/dev/null || true
cp -a "$RESUME_DIR/slurm_launch_stdout.log" "$ASSET_ROOT/logs/resume_slurm_launch_stdout.log" 2>/dev/null || true
cp -a "$ORIG_DIR/slurm_launch_stdout.log" "$ASSET_ROOT/logs/orig_slurm_launch_stdout.log" 2>/dev/null || true

cp -a external/VideoDPO/configs/vc2_dpo "$ASSET_ROOT/configs/vc2_dpo"
cp -a data/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml "$ASSET_ROOT/configs/train_data.absolute.h20.yaml" 2>/dev/null || true

conda env export -p /home/nvme01/conda_envs/videodpo --no-builds \
  | grep -v '^prefix:' > "$ASSET_ROOT/env/videodpo_environment.yml"
/home/nvme01/conda_envs/videodpo/bin/python -m pip freeze \
  > "$ASSET_ROOT/env/videodpo_pip_freeze.txt"

find "$ASSET_ROOT" -type f -printf '%s %P\n' | sort > "$ASSET_ROOT/MANIFEST.txt"

cd /home/nvme01/H20_Video_inpainting_DPO/data/hf_upload
tar -I 'zstd -T0 -3' -cf videodpo_vc2_epoch10_vbench_assets_$(date +%Y%m%d_%H%M%S).tar.zst videodpo_vc2_epoch10_vbench_assets
ls -lh videodpo_vc2_epoch10_vbench_assets_*.tar.zst
```

如果这个包也超过 20G，按 fullmask 方式分片：

```bash
FILE=$(ls -t videodpo_vc2_epoch10_vbench_assets_*.tar.zst | head -1)
PART_DIR="${FILE%.tar.zst}_parts"
rm -rf "$PART_DIR"
mkdir -p "$PART_DIR"

split -b 4G -d -a 3 "$FILE" "$PART_DIR/${FILE}.part-"
sha256sum "$FILE" > "$PART_DIR/${FILE}.sha256"
```

上传：

```bash
source /home/nvme01/clash-for-linux/clash.sh && proxy_on
export HF_HOME=/home/nvme04/hf_cache
unset HF_ENDPOINT

HF=/home/nvme03/workspace/world_model_phys/.conda_envs/phys-main/bin/hf

LOG=/home/nvme04/hf_upload_videodpo_vbench_assets_$(date +%F_%H%M%S).log
nohup "$HF" upload JiaHuang01/New_DPO_data \
  "$PART_DIR" \
  "$PART_DIR" \
  --type dataset \
  --commit-message "Upload VideoDPO VC2 epoch10 VBench assets" \
  > "$LOG" 2>&1 &

echo $! > /home/nvme04/hf_upload_videodpo_vbench_assets.pid
tail -f "$LOG"
```

### 10.2 PAI 下载 VideoDPO VBench 资产

用和 fullmask assets 一样的方式：

1. `unset HF_ENDPOINT`。
2. list repo files，确认新目录出现。
3. `snapshot_download(... allow_patterns=["videodpo_vc2_epoch10_vbench_assets_*_parts/**"])`。
4. `cat part-*` 合并。
5. `sha256sum -c`。
6. `tar -I zstd -xf`。
7. 把 ckpt 放回 `external/VideoDPO/checkpoints/vc2/` 或在 VBench 脚本里显式传 ckpt 路径。

## 11. VBench 后续流程

官方 VideoDPO 复现还缺：

1. 用 final epoch-10-equivalent ckpt 跑 VideoDPO inference。
2. 用 VBench prompt suite 生成视频。
3. 用 VBench 计算指标。
4. 汇总为 `summary.json` 和 `summary.csv`。

已有脚本：

```text
DPO_finetune/scripts/sc_videodpo_vc2_vbench.sbatch
tools/videodpo_prepare_vbench_standard.py
tools/summarize_vbench_results.py
```

PAI 上建议先确认：

```bash
cd /mnt/nas/hj/H20_Video_inpainting_DPO

ls -lh external/VideoDPO/checkpoints/vc2/model.ckpt
ls -lh external/VideoDPO/checkpoints/vc2/ref_model.ckpt
ls -lh /mnt/nas/hj/hf_vbench_assets/videodpo_vc2_epoch10_vbench_assets/checkpoints/vc2/final_epoch10_equiv_last.ckpt
```

然后再跑小规模 inference smoke，确认：

- 能加载 final ckpt。
- 能生成至少 1 个 mp4。
- VBench 环境模型能下载或已缓存。
- 输出视频文件名能被 VBench 标准模式识别。

## 12. H20 Hugging Face 环境注意事项

在 H20 为了用新版 `hf upload`，曾把 `/home/nvme01/conda_envs/videodpo` 里的 `huggingface_hub` 升到了 `1.14.0`。这会和 `transformers==4.25.1` 的要求冲突：

```text
transformers 4.25.1 requires huggingface-hub<1.0,>=0.10.0
```

已经建议修回：

```bash
/home/nvme01/conda_envs/videodpo/bin/python -m pip install 'huggingface_hub==0.25.2' 'hf_xet>=1.4.3'
```

如果只是 HF 上传，最好用另一个独立 HF CLI 环境：

```text
/home/nvme03/workspace/world_model_phys/.conda_envs/phys-main/bin/hf
```

不要为了上传污染训练环境。

## 13. 日常排错规则

### 13.0 GPU 进程名统一显示为 worldmodelphy

不要通过重命名 repo、`DPO_finetune`、`external/VideoDPO`、`vc2_dpo` 等路径来改 GPU 进程名。这些名字是代码和配置语义的一部分，粗暴替换会破坏数据路径、patch、resume、checkpoint 和外部 repo 约定。

当前安全方案：

- repo root 新增 `sitecustomize.py`。
- 主要 GPU 启动脚本默认设置：

```bash
export WORLDMODELPHY_PROCESS_NAME="${WORLDMODELPHY_PROCESS_NAME:-worldmodelphy}"
export PROCESS_TITLE="${PROCESS_TITLE:-${WORLDMODELPHY_PROCESS_NAME}}"
```

- 主要 GPU 启动脚本也会把 `${PROJECT_ROOT}` 加入 `PYTHONPATH`，让 Python、Accelerate、torch distributed worker 自动加载 `sitecustomize.py`。
- 默认显示名是 `worldmodelphy`。
- 若某次实验要改名，可显式传：

```bash
WORLDMODELPHY_PROCESS_NAME=worldmodelphy ...
```

或：

```bash
PROCESS_TITLE=worldmodelphy ...
```

验证：

```bash
WORLDMODELPHY_PROCESS_NAME=worldmodelphy \
PYTHONPATH=/path/to/H20_Video_inpainting_DPO \
python - <<'PY'
from pathlib import Path
print(Path("/proc/self/comm").read_text().strip())
PY
```

期望输出：

```text
worldmodelphy
```

如果某些环境里想让 `ps -ef` 的完整命令行也更干净，可以额外安装：

```bash
python -m pip install setproctitle
```

没有 `setproctitle` 也不会影响训练，`sitecustomize.py` 会退化为 `/proc/self/comm` 方式。

### 13.1 不要全盘 find NAS

NAS 上文件很多，下面这种命令会很慢：

```bash
find /mnt/workspace/hj/nas_hj /mnt/nas/hj -maxdepth 5 ...
```

优先做定点检查：

```bash
ls -lh /mnt/nas/hj/weights/stable-diffusion-v1-5
ls -lh /mnt/nas/hj/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml
```

### 13.2 rsync 报 vanished 不一定致命

H20 训练过程中同步日志时可能出现：

```text
file has vanished
rsync warning: some files vanished before they could be transferred (code 24)
```

这通常是训练进程正在写 `last.ckpt` 或 W&B socket。对大部分静态权重不是致命，但要单独校验关键 ckpt。

### 13.3 rsync 报 mkstemp No such file

出现：

```text
rsync: [receiver] mkstemp ".../.file.tmp" failed: No such file or directory
```

通常说明目标目录结构被移动、删除、软链接混乱，或并行命令在操作同一目录。解决：

- 停掉并行 rsync。
- 确认 `readlink -f`。
- 重新创建目标父目录。
- 只同步一个明确目录。

### 13.4 HF 下载 0 files

可能原因：

- `allow_patterns` 写错。
- HF repo 里是 tar 包，不是展开目录。
- `HF_ENDPOINT=https://hf-mirror.com` mirror 未同步新 commit。
- 上传还没有完成 commit。

处理：

```bash
unset HF_ENDPOINT
python3 - <<'PY'
from huggingface_hub import HfApi
files = HfApi().list_repo_files("JiaHuang01/New_DPO_data", repo_type="dataset")
for f in files:
    print(f)
PY
```

先看真实文件名，再写 `allow_patterns`。

### 13.5 输出太多

8 卡 DDP 会每个 rank 都打印初始化信息和 warnings。训练过程中进度条多行交错是正常的。

真正关注：

- 是否进入 training loop。
- 是否有 `[dpo_diag]`。
- 是否有 `loss` 和 `global/implicit_acc` 更新。
- 是否达到预期停止条件。
- 是否保存 ckpt。

## 14. 最短下一步 checklist

### 14.1 问管理员

- 当前 PAI GPU 型号为什么显示 L20X，不是 H200。
- 推荐个人 NAS/CPFS 目录和 quota。
- HF 官方站点和 mirror 哪个推荐，是否有代理。
- Conda env 是否必须放 NAS。
- 长任务是否推荐 tmux/nohup，是否会抢占。
- CPFS `/mnt/data/pku` 使用成本和清理规则。

### 14.2 PAI fullmask smoke

1. 在 PAI 用官方 HF list repo files，确认分片目录存在。
2. 下载 `fullmask_required_assets_parts_20260515_145007/**`。
3. 合并、sha256 校验、解压。
4. `rsync` 权重到 `/mnt/nas/hj/weights/`。
5. 用 env yaml 创建 `/mnt/nas/hj/conda_envs/diffueraser`。
6. 跑 `pai_videodpo_fullmask_health_check.sh`。
7. 跑 GPU7 2-step smoke。

### 14.3 H20 official VideoDPO VBench

1. 打包 H20 final ckpt、base/ref ckpt、run configs、env manifest。
2. 大包分片上传 HF。
3. PAI 下载并校验。
4. 在 PAI 跑 VideoDPO inference smoke。
5. 跑 VBench。

## 15. 当前结论

- H20 official VideoDPO/VC2 训练已经按原始训练要求结束，final ckpt 可加载。
- fullmask bridge 的代码默认设置已经对齐 official VideoDPO，不应再使用旧的 bf16、512、split-pos-neg 配置。
- PAI 代码和 submodules 已经基本 OK。
- PAI 当前主要缺口是：
  - DiffuEraser conda env。
  - fullmask 必需权重文件落盘和校验。
  - HF 新分片目录在 PAI 侧正确下载。
  - 管理员确认 GPU 型号和存储使用规范。
- official VideoDPO 的 VBench 评测还没做，需要把 H20 final ckpt 和 base/ref ckpt 打包到 PAI 后继续。

## 16. 2026-05-16 PAI Fullmask-DiffuEraser Bridge 最新进展

本节续写 2026-05-16 的完整迁移、修复、smoke、正式训练启动和排错记录。到目前为止，PAI 侧 fullmask-DiffuEraser bridge 已经从“环境/权重/数据缺失”推进到“4 卡正式 epoch-10 训练可启动”的状态。

### 16.1 当前代码 commit

PAI 当前需要拉到以下 commit 或更新：

```text
d69b439 Add VideoDPO epoch mode for fullmask bridge
```

相关新增/修复 commit：

```text
5b5a4e8 Show GPU jobs as worldmodelphy
194ce7c Harden DPO dataset helper imports
afa7058 Resolve repo root for dataset helper fallback
d69b439 Add VideoDPO epoch mode for fullmask bridge
```

各 commit 的作用：

- `5b5a4e8`: 增加 `sitecustomize.py`，让 Python/Accelerate/torch worker 在进程监控里显示指定名字；默认是 `worldmodelphy`。
- `194ce7c`: 增加 `training/common/dataset_imports.py`，避免 PAI 子进程中第三方 `dataset` 包遮蔽 repo 自带 `dataset/file_client.py`。
- `afa7058`: 修复 `PROJECT_ROOT` 在 `train_stage1.py` 中被旧逻辑改成 `repo/training` 后 fallback 找错路径的问题。
- `d69b439`: 增加 VideoDPO-compatible epoch mode，fullmask bridge 默认用 `MAX_EPOCHS=10` 推导训练步数；同时支持关闭每 step 的 DPO 大表输出。

PAI 拉取方式：

```bash
cd /mnt/nas/hj/H20_Video_inpainting_DPO

for i in 1 2 3 4 5; do
  git pull --ff-only origin main && break
  sleep 5
done

git rev-parse --short HEAD
```

如果 GitHub 报：

```text
GnuTLS recv error (-110): The TLS connection was non-properly terminated
```

则只是网络抖动，重试即可。不要在没拉到新 commit 时重跑旧脚本。

### 16.2 PAI 资源和路径实际状态

PAI 当前工作目录：

```text
/mnt/nas/hj/H20_Video_inpainting_DPO
```

该路径与之前的：

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO
```

指向同一个 NAS 项目位置，后续统一使用 `/mnt/nas/hj/H20_Video_inpainting_DPO`。

PAI health check 看到的 GPU：

```text
8x NVIDIA L20X
```

管理员手册写的是：

```text
lingbot_fast_multi: 8x H200
```

这两者不一致，仍需要向管理员确认。但当前 L20X 机器可以正常完成 fullmask health check、dataset smoke、1 卡训练 smoke、4 卡 DDP 启动。

### 16.3 Fullmask 必需资产已经从 H20 成功传到 PAI

H20 上打包的 fullmask 必需资产：

```text
/home/nvme01/H20_Video_inpainting_DPO/data/hf_upload/fullmask_required_assets_20260515_145007.tar.zst
```

内容包括：

- `weights/stable-diffusion-v1-5/model_index.json`
- `weights/stable-diffusion-v1-5/tokenizer`
- `weights/stable-diffusion-v1-5/scheduler`
- `weights/stable-diffusion-v1-5/text_encoder`
- `weights/stable-diffusion-v1-5/unet`
- `weights/sd-vae-ft-mse`
- `weights/diffuEraser/converted_weights_step48000`
- `env/diffueraser_environment.yml`
- `env/diffueraser_pip_freeze.txt`
- `MANIFEST.txt`

HF 单文件和分片上传/下载曾遇到 Xet/mirror/cas-bridge 很慢或 mirror 不同步的问题。最终采用 H20 到 PAI 服务器间 `rsync`，速度约 11 MB/s，成功传输：

```text
/mnt/nas/hj/hf_New_DPO_data/fullmask_required_assets_parts_20260515_145007/
```

合并后的 PAI 包：

```text
/mnt/nas/hj/hf_New_DPO_data/fullmask_required_assets_20260515_145007.tar.zst
```

校验通过：

```text
fullmask_required_assets_20260515_145007.tar.zst: OK
```

解压位置：

```text
/mnt/nas/hj/hf_fullmask_assets/fullmask_required_assets
```

权重落盘位置：

```text
/mnt/nas/hj/weights
```

关键文件已确认存在：

```text
/mnt/nas/hj/weights/stable-diffusion-v1-5/model_index.json
/mnt/nas/hj/weights/stable-diffusion-v1-5/tokenizer/vocab.json
/mnt/nas/hj/weights/sd-vae-ft-mse/config.json
/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000/unet_main/config.json
/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000/brushnet/config.json
```

完整同步命令参考：

```bash
mkdir -p /mnt/nas/hj/weights

rsync -ahP \
  /mnt/nas/hj/hf_fullmask_assets/fullmask_required_assets/weights/ \
  /mnt/nas/hj/weights/
```

### 16.4 PAI DiffuEraser 环境已经建立

Conda 环境位置：

```text
/mnt/nas/hj/conda_envs/diffueraser
```

原始 H20 导出的 env yaml 里包含：

```text
clip==1.0
```

PAI 直接 `conda env create` 时从 PyPI 找不到该版本：

```text
ERROR: Could not find a version that satisfies the requirement clip==1.0
```

处理方式：

1. 从 env yaml 中删除 `- clip==1.0`。
2. 用修正后的 yaml 创建环境。
3. GitHub 访问失败时，不通过 `pip install git+https://github.com/openai/CLIP.git`，而是从 H20 直接 rsync `clip/` 包目录。

创建环境的实际命令：

```bash
ENV=/mnt/nas/hj/conda_envs/diffueraser
SRC=/mnt/nas/hj/hf_fullmask_assets/fullmask_required_assets/env/diffueraser_environment.yml
FIX=/mnt/nas/hj/hf_fullmask_assets/fullmask_required_assets/env/diffueraser_environment.pai.yml

python3 - <<PY
from pathlib import Path
src = Path("$SRC")
dst = Path("$FIX")
lines = src.read_text().splitlines()
lines = [x for x in lines if x.strip() != "- clip==1.0"]
dst.write_text("\n".join(lines) + "\n")
print(dst)
PY

conda env remove -p "$ENV" -y || true
rm -rf "$ENV"
conda env create -p "$ENV" -f "$FIX"
```

从 H20 拷贝 CLIP：

```bash
ENV=/mnt/nas/hj/conda_envs/diffueraser
SITE=$("$ENV/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)

rsync -ahP --partial --append-verify \
  -e "ssh -p 22" \
  ubuntu@27.190.15.128:/home/nvme01/conda_envs/diffueraser/lib/python3.10/site-packages/clip/ \
  "$SITE/clip/"
```

`clip-1.0.dist-info/` 拷贝时曾因为 SSH reset 失败，但不是训练阻塞项。实际 import 已通过：

```text
torch 2.3.1+cu121
cuda_available True
cuda_device_count 1
clip /mnt/nas/hj/conda_envs/diffueraser/lib/python3.10/site-packages/clip/__init__.py
```

为了进程名显示，还安装了：

```bash
/mnt/nas/hj/conda_envs/diffueraser/bin/python -m pip install setproctitle
```

### 16.5 PAI VideoDPO 数据路径修复

PAI 上 VC2 数据实际 root：

```text
/mnt/nas/hj/data/external/hf/vidpro10k-vc2-dataset/_extracted/home/liurt/liurt_data/haoyu/dataset/vidpro10k-vc2-dataset
```

其中：

```text
metadata.json = 20000 video records
pair.json     = 10000 winner/loser preference pairs
```

注意：`6250` 不是视频数。它是 optimizer steps。当前数据含义如下：

- `metadata=20000`: 20000 个视频记录。
- `pair=10000`: 10000 个偏好对。
- 每个 pair 包含 winner/loser 两个视频索引。

VideoDPO-compatible 训练步数推导：

```text
num_pairs = 10000
per_device_batch = 1
num_gpus = 8
grad_accum = 2
effective_batch_pairs = 1 * 8 * 2 = 16 pairs / optimizer step
10 epoch = 10000 * 10 / 16 = 6250 optimizer steps
```

如果使用 4 卡，为了保持 effective batch 仍为 16，应使用：

```text
num_gpus = 4
grad_accum = 4
effective_batch_pairs = 1 * 4 * 4 = 16 pairs / optimizer step
10 epoch = 10000 * 10 / 16 = 6250 optimizer steps
```

这就是为什么 4 卡时必须 `GRAD_ACCUM=4`，而不是旧命令的 `GRAD_ACCUM=2`。旧命令 `NUM_GPUS=4, GRAD_ACCUM=2, MAX_STEPS=6250` 的 effective batch 是 8，只等价约 5 epoch：

```text
6250 * 8 / 10000 = 5 epoch
```

PAI 需要的 train data yaml：

```text
/mnt/nas/hj/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.pai.yaml
```

内容：

```yaml
META:
- /mnt/nas/hj/data/external/hf/vidpro10k-vc2-dataset/_extracted/home/liurt/liurt_data/haoyu/dataset/vidpro10k-vc2-dataset
```

最初 `metadata.json` 内部的 `clip_path` 仍指向 H20 绝对路径：

```text
/home/nvme01/H20_Video_inpainting_DPO/data/external/hf/...
```

导致 health check 报：

```text
first_clip exists=False
VideoDPO data validation failed
full-mask dataset smoke failed
```

已用下面脚本重写 `metadata.json` 中的 H20 prefix 到 PAI prefix：

```bash
export DATA_ROOT=/mnt/nas/hj/data/external/hf/vidpro10k-vc2-dataset/_extracted/home/liurt/liurt_data/haoyu/dataset/vidpro10k-vc2-dataset
export OLD_PREFIX=/home/nvme01/H20_Video_inpainting_DPO/data/external/hf/vidpro10k-vc2-dataset/_extracted/home/liurt/liurt_data/haoyu/dataset/vidpro10k-vc2-dataset
export NEW_PREFIX="$DATA_ROOT"

python3 - <<'PY'
import json, os, shutil
from pathlib import Path

root = Path(os.environ["DATA_ROOT"])
old = os.environ["OLD_PREFIX"]
new = os.environ["NEW_PREFIX"]

def rewrite(x):
    if isinstance(x, str):
        return x.replace(old, new)
    if isinstance(x, list):
        return [rewrite(v) for v in x]
    if isinstance(x, dict):
        return {k: rewrite(v) for k, v in x.items()}
    return x

for name in ["metadata.json", "pair.json"]:
    p = root / name
    data = json.load(p.open())
    fixed = rewrite(data)
    if fixed != data:
        bak = root / f"{name}.h20_abs.bak"
        if not bak.exists():
            shutil.copy2(p, bak)
        tmp = root / f"{name}.tmp"
        json.dump(fixed, tmp.open("w"), ensure_ascii=False)
        tmp.replace(p)
        print("[rewrote]", p, "backup=", bak)
    else:
        print("[no change]", p)
PY
```

结果：

```text
[rewrote] metadata.json backup=metadata.json.h20_abs.bak
[no change] pair.json
```

修复后 first clip：

```text
/mnt/nas/hj/data/external/hf/vidpro10k-vc2-dataset/.../winvideos/000000.mp4 exists=True
```

### 16.6 PAI health check 已通过

最终 health check 命令：

```bash
cd /mnt/nas/hj/H20_Video_inpainting_DPO

VC2_YAML=/mnt/nas/hj/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.pai.yaml

DIFFUERASER_ENV=/mnt/nas/hj/conda_envs/diffueraser \
CONDA_ENV=/mnt/nas/hj/conda_envs/diffueraser \
WEIGHTS_DIR=/mnt/nas/hj/weights \
DPO_DATA_ROOT="$VC2_YAML" \
bash DPO_finetune/scripts/pai_videodpo_fullmask_health_check.sh 2>&1 | tee /tmp/pai_health_check.log
```

结果：

```text
fails=0 warns=6
[SUMMARY] health check passed
```

关键 OK 项：

- Git repo: OK
- working tree clean on PAI: OK
- VideoDPO submodule layout: OK
- VBench layout: OK
- DiffuEraser env python: OK
- torch/cuda/decord/diffusers/transformers import: OK
- CUDA available: OK
- weights root: OK
- SD1.5 tokenizer/text_encoder/unet: OK
- sd-vae-ft-mse: OK
- DiffuEraser unet_main/brushnet: OK
- VC2 metadata/pair: OK
- first clip exists: OK
- full-mask dataset smoke: OK

Dataset smoke 关键输出：

```text
VideoDPOFullMaskDiffuEraserDataset: pairs=10000 roots=1 resolution=[320,512] nframes=16 frame_stride=1
[smoke] len=10000 index=0
[smoke] pixel_values_pos: shape=(16, 3, 320, 512) dtype=torch.float32 min=-1.0000 max=1.0000
[smoke] pixel_values_neg: shape=(16, 3, 320, 512) dtype=torch.float32 min=-1.0000 max=1.0000
[smoke] conditioning_pixel_values: shape=(16, 3, 320, 512) dtype=torch.float32 min=-1.0000 max=-1.0000
[smoke] masks: shape=(16, 1, 320, 512) dtype=torch.float32 min=0.0000 max=0.0000
[smoke] input_ids: shape=(77,) dtype=torch.int64
```

剩余 warning 不阻塞 fullmask 训练：

- `rg not installed`: 只影响 conflict-marker scan。
- `VideoDPO env python not found`: 只影响 official VideoDPO/VBench，不影响 fullmask DiffuEraser。
- official `model.ckpt/ref_model.ckpt` missing: 只影响 official VideoDPO/VBench，不影响 fullmask DiffuEraser stage1。

### 16.7 Prompt 输入已确认可用

这条 fullmask-DiffuEraser bridge 不是无 prompt inpainting。它已经支持 VideoDPO prompt/caption 输入。

证据：

- dataset smoke 有 `input_ids: shape=(77,)`。
- 训练代码实际执行：

```python
encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]
```

当前数据流：

```text
winner video -> pixel_values_pos
loser video  -> pixel_values_neg
pair caption 或 metadata caption -> tokenizer -> input_ids
input_ids -> text_encoder -> encoder_hidden_states
full-mask conditioning image -> conditioning_pixel_values
full-mask tensor -> masks
encoder_hidden_states + noisy latents + brushnet_cond -> DiffuEraser policy/ref forward
```

也就是说，VideoDPO 的文本 prompt 已经进入 SD1.5 text encoder，并参与 DiffuEraser `unet/brushnet` cross-attention。这个 bridge 的实验定义是：

```text
保持 VideoDPO 数据和 preference task 不变，只把模型侧换成 fullmask DiffuEraser。
```

### 16.8 1 卡 2-step smoke 已通过

命令：

```bash
cd /mnt/nas/hj/H20_Video_inpainting_DPO

VC2_YAML=/mnt/nas/hj/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.pai.yaml

CUDA_VISIBLE_DEVICES=7 \
DIFFUERASER_ENV=/mnt/nas/hj/conda_envs/diffueraser \
CONDA_ENV=/mnt/nas/hj/conda_envs/diffueraser \
PROJECT_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO \
PROJECT_HOME=/mnt/nas/hj \
PROJECT_DEV=/mnt/nas/hj \
PROJECT_DATA=/mnt/nas/hj/H20_Video_inpainting_DPO \
DATA=/mnt/nas/hj/H20_Video_inpainting_DPO \
WEIGHTS_DIR=/mnt/nas/hj/weights \
DPO_DATA_ROOT="$VC2_YAML" \
RUN_NAME=pai-gpu7-worldmodelphy-fullmask-diffueraser-smoke-$(date +%Y%m%d_%H%M%S) \
NUM_GPUS=1 \
MAX_STEPS=2 \
CKPT_STEPS=999999 \
VAL_STEPS=999999 \
LOGGING_STEPS=1 \
REPORT_TO=none \
bash DPO_finetune/scripts/sc_videodpo_fullmask_diffueraser_stage1.sbatch
```

启动设置确认：

```text
precision=no
train_size=320x512
split_pos_neg=0
beta=5000
lr=6e-6
```

训练成功点：

- 成功加载 SD1.5、VAE、DiffuEraser policy/ref。
- 成功从 `UNetMotionModel` 提取 2D UNet 权重。
- dataset 成功构造 `pairs=10000`。
- DPO Stage1 training loop 成功进入。
- step 1/2 和 step 2/2 完成。
- DGR grad norm > 0，反向传播和 optimizer step 都活着。
- 保存了：

```text
.../last_weights/unet_main/diffusion_pytorch_model.safetensors
.../last_weights/brushnet/diffusion_pytorch_model.safetensors
```

1 卡 smoke 结束时曾出现大量：

```text
OSError: [Errno 16] Device or resource busy: '.nfs...'
```

原因是 DataLoader/multiprocessing 在 NAS 上清理临时目录时触发 NFS 临时文件 busy。不是训练失败。后续正式训练应设置：

```bash
TMPDIR=/tmp/hj_worldmodel_tmp
```

把 multiprocessing 临时目录放在本机 `/tmp`，避免 NAS NFS 清理噪声。

### 16.9 dataset.file_client import 问题和修复

PAI 1 卡 smoke 首次进入训练前失败：

```text
ModuleNotFoundError: No module named 'dataset.file_client'
```

手动测试：

```bash
PYTHONPATH="$PWD:${PYTHONPATH:-}" \
/mnt/nas/hj/conda_envs/diffueraser/bin/python - <<'PY'
import dataset
print(dataset.__file__)
from dataset.file_client import FileClient
from dataset.img_util import imfrombytes
print("dataset imports OK")
PY
```

是 OK 的，但 Accelerate 子进程仍失败。说明训练子进程内 `dataset` 包名存在 shadowing 或 `sys.path` 顺序问题。

修复：

- 新增 `training/common/dataset_imports.py`。
- DPO stage1/stage2 不再直接依赖 `from dataset.file_client import FileClient`。
- 先尝试正常 import。
- 如果失败，就从 repo root 的 `dataset/file_client.py` 和 `dataset/img_util.py` 文件路径直接加载。
- 修复 `PROJECT_ROOT` 被旧代码改成 `repo/training` 后 fallback 找错路径的问题。

相关 commit：

```text
194ce7c Harden DPO dataset helper imports
afa7058 Resolve repo root for dataset helper fallback
```

### 16.10 4 卡旧 run 已手动中断

曾启动 4 卡旧 run：

```text
CUDA_VISIBLE_DEVICES=4,5,6,7
NUM_GPUS=4
GRAD_ACCUM=2
MAX_STEPS=6250
LOGGING_STEPS=1
```

该 run 成功跑起来，说明 4 卡 DDP/NCCL/model/dataset 都可用。它跑到大约：

```text
45/6250
```

然后用户手动 `Ctrl-C` 中断。日志中的：

```text
KeyboardInterrupt
NCCL Abort COMPLETE
```

是正常的手动中断清理，不是模型崩溃。

停掉该 run 是正确的，因为它不是 strict VideoDPO epoch-10 对齐：

```text
4 cards * batch 1 * grad_accum 2 = effective batch 8
6250 steps * 8 / 10000 pairs = 5 epoch
```

### 16.11 新版 official-compatible epoch=10 启动方式

从 commit `d69b439` 起，fullmask wrapper 支持：

```text
MAX_EPOCHS=10
MAX_STEPS unset/empty
```

脚本会向训练入口传：

```text
--num_train_epochs 10
```

而不是固定传：

```text
--max_train_steps 6250
```

训练脚本内部再根据 dataloader 和 effective batch 推导实际 total optimization steps。

正式 4 卡启动前先确认无残留：

```bash
nvidia-smi
ps -ef | grep -E 'train_stage1|accelerate|lingbot-world' | grep -v grep
```

推荐启动命令，保留 diagnostics 但不每 step 刷屏：

```bash
cd /mnt/nas/hj/H20_Video_inpainting_DPO
mkdir -p logs/fullmask_runs /tmp/hj_worldmodel_tmp

VC2_YAML=/mnt/nas/hj/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.pai.yaml
RUN_NAME=lingbot-world-model-fullmask-videodpo-epoch10-gpu4-7-$(date +%Y%m%d_%H%M%S)
LOG=logs/fullmask_runs/${RUN_NAME}.log

unset MAX_STEPS

CUDA_VISIBLE_DEVICES=4,5,6,7 \
WORLDMODELPHY_PROCESS_NAME=lingbot-world-model \
PROCESS_TITLE=lingbot-world-model \
NCCL_DEBUG=WARN \
PYTHONWARNINGS=ignore \
TMPDIR=/tmp/hj_worldmodel_tmp \
DIFFUERASER_ENV=/mnt/nas/hj/conda_envs/diffueraser \
CONDA_ENV=/mnt/nas/hj/conda_envs/diffueraser \
PROJECT_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO \
PROJECT_HOME=/mnt/nas/hj \
PROJECT_DEV=/mnt/nas/hj \
PROJECT_DATA=/mnt/nas/hj/H20_Video_inpainting_DPO \
DATA=/mnt/nas/hj/H20_Video_inpainting_DPO \
WEIGHTS_DIR=/mnt/nas/hj/weights \
DPO_DATA_ROOT="$VC2_YAML" \
RUN_NAME="$RUN_NAME" \
NUM_GPUS=4 \
GRAD_ACCUM=4 \
MAX_EPOCHS=10 \
CKPT_STEPS=499 \
VAL_STEPS=999999 \
LOGGING_STEPS=499 \
DPO_DIAGNOSTICS=1 \
REPORT_TO=none \
bash DPO_finetune/scripts/sc_videodpo_fullmask_diffueraser_stage1.sbatch 2>&1 | tee "$LOG"
```

启动日志必须确认：

```text
max_epochs=10
max_steps=auto
diagnostics=1
GPUs: 4
Gradient Accumulation steps = 4
Total train batch size = 16
Num Epochs = 10
```

如果只想屏幕上更干净，仍然保存完整 log，可以继续用 `tee "$LOG"`。如果完全不想屏幕显示大表，可设置：

```text
DPO_DIAGNOSTICS=0
```

但当前建议正式实验保留：

```text
DPO_DIAGNOSTICS=1
LOGGING_STEPS=499
```

这样只在 step 1、499、998、1497、... 打一次 DPO diagnostics，大致和 checkpoint 节奏对齐，便于复现实验审查。

### 16.12 DPO diagnostics 保存位置

H20 official VideoDPO 复现中的 `dpo_diag` 查找方式：

```bash
for RUN in \
h20-vc2-dpo-official-full-gpu0-7_20260514_035204 \
h20-vc2-dpo-official-full-gpu0-7_resume-step2495_20260514_195149
do
  RUN_DIR=/home/nvme01/H20_Video_inpainting_DPO/logs/videodpo_vc2_dpo/$RUN
  echo "===== $RUN ====="
  grep -R "\[dpo_diag\]" "$RUN_DIR/slurm_launch_stdout.log" "$RUN_DIR/loginfo" 2>/dev/null | tail -30
done
```

本次 DiffuEraser fullmask bridge 的 diagnostics 不是 `[dpo_diag]` 格式，而是大表格式：

```text
DPO Diagnostics @ Step ...
```

保存位置有两类：

1. 终端 tee 总日志：

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/fullmask_runs/${RUN_NAME}.log
```

2. 每个 run 的 output dir 下 console/logging 文件：

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/<timestamp>_<RUN_NAME>/
```

建议每次正式 run 都用：

```bash
LOG=logs/fullmask_runs/${RUN_NAME}.log
... bash DPO_finetune/scripts/sc_videodpo_fullmask_diffueraser_stage1.sbatch 2>&1 | tee "$LOG"
```

后续快速查看 diagnostics：

```bash
grep -n "DPO Diagnostics @ Step" "$LOG" | tail -20
grep -n "DGR (grad norm)" "$LOG" | tail -20
grep -n "Implicit Acc" "$LOG" | tail -20
```

查看最近 run：

```bash
ls -td /mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/* | head
ls -lt /mnt/nas/hj/H20_Video_inpainting_DPO/logs/fullmask_runs | head
```

### 16.13 输出太多的原因和控制方法

输出很多的原因有四个：

1. `LOGGING_STEPS=1`：每 step 打完整 DPO diagnostics 大表。
2. `NCCL_DEBUG=INFO`：DDP 初始化时打印大量网络和 ring/tree 拓扑。
3. 多卡每个 rank 都会打印 warning 或初始化信息。
4. `TRANSFORMERS_CACHE` deprecation、config missing fields 等 warning。

控制方式：

```bash
NCCL_DEBUG=WARN
PYTHONWARNINGS=ignore
LOGGING_STEPS=499
DPO_DIAGNOSTICS=1
```

如果只要进度条，完全关闭 DPO 大表：

```bash
DPO_DIAGNOSTICS=0
LOGGING_STEPS=999999
```

但正式复现实验建议不要完全关掉 diagnostics，至少每个 checkpoint 附近留一份。

### 16.14 GPU 进程名显示策略

用户要求不管谁 `nvidia-smi`，都尽量显示：

```text
lingbot-world-model
```

启动时传：

```bash
WORLDMODELPHY_PROCESS_NAME=lingbot-world-model
PROCESS_TITLE=lingbot-world-model
```

检查：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
ps -eo pid,comm,args | grep -E 'lingbot-world|train_stage1|accelerate' | grep -v grep
```

注意系统限制：

- Linux `/proc/<pid>/comm` 最多 15 个可见字符。
- `lingbot-world-model` 长度超过 15。
- 某些 `nvidia-smi` 版本可能显示成：

```text
lingbot-world-m
```

- 如果 `nvidia-smi` 读取完整 argv 或 setproctitle，可能显示完整 `lingbot-world-model`。
- 如果必须保证不截断，名字应换成 15 字符以内，例如：

```text
lingbot-world
lingbotwm
```

当前保留用户指定的完整名 `lingbot-world-model`。

### 16.15 当前 fullmask bridge 和 official VideoDPO 的对齐关系

必须保持：

```text
train size: 320x512
video length: 16
frame stride: 1
per device batch: 1
effective batch: 16 pairs / optimizer step
lr: 6e-6
beta: 5000
precision: fp32 / mixed_precision=no
checkpoint interval: 499
epoch target: 10
```

4 卡对齐方案：

```text
NUM_GPUS=4
GRAD_ACCUM=4
MAX_EPOCHS=10
```

8 卡对齐方案：

```text
NUM_GPUS=8
GRAD_ACCUM=2
MAX_EPOCHS=10
```

不应再使用：

```text
MIXED_PRECISION=bf16
SPLIT_POS_NEG_FORWARD=1
RESOLUTION-only 512 square training
```

当前 wrapper 会默认：

```text
TRAIN_HEIGHT=320
TRAIN_WIDTH=512
MIXED_PRECISION=no
SPLIT_POS_NEG_FORWARD=0
BETA_DPO=5000
LR=6e-6
CKPT_STEPS=499
NUM_WORKERS=16
MAX_EPOCHS=10
```

### 16.16 当前剩余风险

- 当前 PAI GPU 显示 L20X，不是管理员手册中的 H200。训练已可跑，但性能和实验描述里应写清楚实际 GPU。
- 4 卡正式 run 还需要重新用 `d69b439` 的 epoch mode 启动，旧 45-step run 已中断，不应作为正式结果。
- `nvidia-smi` 进程名可能因 Linux 15 字符 comm 限制被截断。
- `NCCL_DEBUG=INFO` 会造成日志极大，正式 run 应用 `NCCL_DEBUG=WARN`。
- `DPO_DIAGNOSTICS=0` 虽然屏幕干净，但会少掉可审计诊断；正式建议 `DPO_DIAGNOSTICS=1, LOGGING_STEPS=499`。
- official VideoDPO VBench 仍未完成。
- PAI official VideoDPO/VBench 所需的 `model.ckpt/ref_model.ckpt/final_epoch10_equiv_last.ckpt` 仍需要单独确认/同步。

## 17. 2026-05-16 下一步 checklist

1. 确认旧 4 卡 run 没有残留：

```bash
nvidia-smi
ps -ef | grep -E 'train_stage1|accelerate|lingbot-world' | grep -v grep
```

2. 确认 PAI 已经是 `d69b439` 或更新：

```bash
cd /mnt/nas/hj/H20_Video_inpainting_DPO
git rev-parse --short HEAD
```

3. 用 `NUM_GPUS=4, GRAD_ACCUM=4, MAX_EPOCHS=10` 重新启动 fullmask formal run。

4. 启动后确认日志：

```text
max_epochs=10
max_steps=auto
Total train batch size = 16
Num Epochs = 10
```

5. 确认 GPU 进程名：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
ps -eo pid,comm,args | grep -E 'lingbot-world|train_stage1|accelerate' | grep -v grep
```

6. 训练中每隔一段时间查看：

```bash
tail -n 80 "$LOG"
grep -n "DPO Diagnostics @ Step" "$LOG" | tail
grep -n "DGR (grad norm)" "$LOG" | tail
```

7. checkpoint 检查：

```bash
ls -lh /mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/<RUN_DIR>/checkpoint-*
ls -lh /mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/<RUN_DIR>/last_weights
```

8. 训练完成后，把 run dir、tee log、health check log、code commit hash、env yaml、weights manifest 汇总归档。

## 18. 2026-05-16/17 续写：official VideoDPO VBench、PAI 环境迁移、fullmask DiffuEraser DPO 训练完成与评测规划

本节只追加目前已经发生和已经确认的内容，不覆盖前文。时间戳以 PAI/H20 终端中显示的 CST 为准；本地 PRD 写入时间可能显示为欧洲时区。

### 18.1 本轮目标

本轮从 fullmask DiffuEraser DPO 训练可跑，推进到两个层面的评测闭环：

1. 复现 official VideoDPO 的 VC2 baseline vs VC2-DPO 评测：
   - 使用官方 VC2 checkpoint `model.ckpt` 作为 baseline。
   - 使用 H20 复现 VideoDPO DPO 训练得到的 `last.ckpt` 作为 DPO model。
   - 用 VBench full standard prompt suite 做定量评测。
   - 额外抽 30 个随机 VBench prompt 做定性 side-by-side 二合一视频。

2. 对 PAI 上完成的 fullmask-DiffuEraser-VideoDPO finetune 做完全同构评测：
   - baseline 是原始 DiffuEraser converted weights。
   - DPO model 是 PAI 5 epoch fullmask-DPO 训练产物 `last_weights`。
   - 同样用 full VBench standard prompt suite 做定量评测。
   - 同样抽 30 个随机 VBench prompt 做二合一定性视频。

最终应产出四类核心结果：

```text
1. ordinary VideoDPO: VC2-base vs VC2-DPO 的二合一定性视频目录
2. fullmask DiffuEraser: DiffuEraser-base vs FullMask-DPO 的二合一定性视频目录
3. ordinary VideoDPO: paper-style VBench table
4. fullmask DiffuEraser: paper-style VBench table
```

### 18.2 重要概念澄清：DiffuEraser fullmask 不是 ProPainter prior

这轮实验里反复确认了一个容易混淆的点：

```text
fullmask-DiffuEraser-VideoDPO 实验不能加入 ProPainter prior。
```

当前 fullmask bridge 的输入设置是：

```text
prompt: 来自 VideoDPO / VBench prompt
masked images: 全黑空图
masks: 全 mask
weights: DiffuEraser unet_main + brushnet
```

因此它不是使用首帧条件的 image-to-video，也不是 ProPainter 修复后再 diffusion 的 prior pipeline。它本质上是把 DiffuEraser 的 video inpainting architecture 放在 full-mask blank-conditioning 设置下使用，让 prompt 和模型本身完成生成。论文图 2 中输入是 mask 和 masked image，而不是首帧；fullmask 评测也保持这一点。

baseline 对应：

```text
/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000
```

DPO finetuned model 对应：

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260516_024051_lingbot-world-model-fullmask-videodpo-epoch5-gpu4-7-20260516_104046/last_weights
```

baseline 和 DPO 必须用完全一样的 fullmask blank-conditioning VBench setting 进行比较。

### 18.3 Git 代码演进与关键 commit

PAI 一开始多次 `git pull` 失败，出现过：

```text
curl 52 Empty reply from server
curl 56 GnuTLS recv error (-110)
curl 28 Failed to connect to github.com port 443
fatal: expected acknowledgments
fatal: expected flush after ref listing
```

后来在 PAI 上直接 `git pull --ff-only origin main` 成功，从 `904eda7` 更新到 `65eefe0`。之后本地又追加了 `9160ab4`。

当前与本轮评测强相关的 commits：

```text
d69b439 Add VideoDPO epoch mode for fullmask bridge
10db180 Reduce DPO progress bar noise
94af176 Match VideoDPO DPO diagnostics line
cdbc7bc Use explicit Python for VideoDPO VBench
589d1f8 Patch VideoDPO savefps for VBench smoke
904eda7 Patch VideoDPO writer for PyAV compatibility
f5c56db Add paper-style VBench table export
ea946f2 Add multi GPU qualitative VBench smoke
65eefe0 Add VBench named side-by-side helper
9160ab4 Add prompt sampling helper
```

几个新增/修改工具的职责：

```text
DPO_finetune/scripts/sc_videodpo_vc2_vbench.sbatch
  - official VideoDPO VC2/DPO VBench wrapper。
  - 使用显式 CONDA_ENV python，避免拿错 phys-main 或 base python。
  - runtime patch VideoDPO inference.py，使其支持 VIDEODPO_INFER_RANK/VIDEODPO_INFER_NUM_GPUS。
  - runtime patch PyAV/torchvision writer 问题，改用 imageio/cv2 fallback。
  - 支持 PROMPT_RANDOM_LIMIT=30 做定性 smoke。
  - 支持 NUM_INFER_GPUS 多 GPU 切分 prompt。
  - RUN_VBENCH=1 时自动产出 VBench summary 和 paper-style table。

tools/vbench_make_paper_table.py
  - 将 VBench summary.json 汇总成 paper-style md/csv/tex。
  - ordinary VideoDPO 默认可生成类似论文表格的列：
    Backbone, Model, VBench Total (%), VBench Quality (%), VBench Semantics (%), HPS (V), PickScore。
  - HPS/PickScore 当前 VBench wrapper 不计算，因此输出为 "-"; 不能伪造。

tools/videodpo_make_side_by_side.py
  - ordinary VideoDPO raw output 形如 0001.mp4, 0002.mp4。
  - 该工具将 vc2_base 与 vc2_dpo 对应 index 的视频左右拼接，并写明标签。

tools/vbench_named_side_by_side.py
  - 用于 VBench-standard named folder。
  - 匹配 `<prompt>-<sample_index>.mp4`。
  - 用于 fullmask DiffuEraser baseline vs DPO 的二合一定性视频。

tools/sample_prompts.py
  - 固定 seed 从 prompt txt 中抽样。
  - 注意：只用于定性 smoke，不能用于定量 VBench。
```

### 18.4 H20 official VideoDPO DPO diagnostics log 的真实位置与图表

H20 上 official VideoDPO DPO 复现日志真实来源包括：

```text
/home/nvme01/H20_Video_inpainting_DPO/logs/videodpo_vc2_dpo/h20-vc2-dpo-official-full-gpu0-7_resume-step2495_20260514_195149/slurm_launch_stdout.log

/home/nvme01/H20_Video_inpainting_DPO/logs/videodpo_vc2_dpo/h20-vc2-dpo-official-full-gpu0-7_resume-step2495_20260514_195149/loginfo/log_0:2026-05-14T19-52-01.txt
...
/home/nvme01/H20_Video_inpainting_DPO/logs/videodpo_vc2_dpo/h20-vc2-dpo-official-full-gpu0-7_resume-step2495_20260514_195149/loginfo/log_7:2026-05-14T19-52-01.txt
```

曾打包到：

```text
/home/nvme01/H20_Video_inpainting_DPO/data/hf_upload/videodpo_official_diag_logs_20260516_105954.tar.zst
/home/nvme01/H20_Video_inpainting_DPO/data/hf_upload/videodpo_official_diag_logs_20260516_105954.tar.zst.sha256
```

本地整理出的 DPO diagnostics 图和数据位于：

```text
/home/hj/Video_inpainting_DPO/videodpo-vc2-dpo-diag-log/dpo_diag_lines.txt
/home/hj/Video_inpainting_DPO/videodpo-vc2-dpo-diag-log/dpo_diag_metrics_dedup.csv
/home/hj/Video_inpainting_DPO/videodpo-vc2-dpo-diag-log/dpo_diag_overview.png
/home/hj/Video_inpainting_DPO/videodpo-vc2-dpo-diag-log/dpo_diag_overview.pdf
/home/hj/Video_inpainting_DPO/videodpo-vc2-dpo-diag-log/dpo_diag_key_signals.png
```

典型 `[dpo_diag]` 行包含：

```text
global_step
implicit_acc
implicit_acc_count
inside_term_mean/min/max
loser_dominant_ratio
dpo_loss
mse_w/ref_mse_w
mse_l/ref_mse_l
win_gap
lose_gap
reward_margin
sigma_term
kl_divergence
```

official VideoDPO diagnostics 的保存方式是写入 `slurm_launch_stdout.log` 和 `loginfo/log_*`。本项目 fullmask DPO 也应保持同类审计日志，而不是只在屏幕显示。

### 18.5 H20 -> PAI 资产迁移：权重与环境

#### 18.5.1 权重迁移

PAI 审查最初发现缺少 official VC2 checkpoint 和 VideoDPO 环境：

```text
[MISS] VC2 base model.ckpt
[MISS] VC2 ref_model.ckpt
[MISS] python env: /mnt/nas/hj/conda_envs/videodpo/bin/python
[MISS] pytorch_lightning
[MISS] kornia
```

从 H20 `ubuntu@27.190.15.128` 传到 PAI 后，关键权重已就位：

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/external/VideoDPO/checkpoints/vc2/model.ckpt
  size: 6.9G

/mnt/nas/hj/H20_Video_inpainting_DPO/external/VideoDPO/checkpoints/vc2/ref_model.ckpt
  size: 5.3G

/mnt/nas/hj/h20_vbench_assets/checkpoints/vc2_dpo_last.ckpt
  size: 18G
```

总传输量约 30G，网络上大约 11MB/s。过程中 `ref_model.ckpt` 曾因 SSH broken pipe / reset 失败，后来使用 `rsync --partial --append-verify` 重试成功。

可靠传输命令模式：

```bash
cd /mnt/nas/hj/H20_Video_inpainting_DPO
H20=ubuntu@27.190.15.128
SSH='ssh -p 22 -o ServerAliveInterval=30 -o ServerAliveCountMax=120 -o ConnectTimeout=30'

mkdir -p external/VideoDPO/checkpoints/vc2
mkdir -p /mnt/nas/hj/h20_vbench_assets/checkpoints

rsync -ahP -L --partial --append-verify -e "$SSH" \
  "$H20":/home/nvme01/H20_Video_inpainting_DPO/external/VideoDPO/checkpoints/vc2/model.ckpt \
  /mnt/nas/hj/H20_Video_inpainting_DPO/external/VideoDPO/checkpoints/vc2/model.ckpt

rsync -ahP -L --partial --append-verify -e "$SSH" \
  "$H20":/home/nvme01/H20_Video_inpainting_DPO/external/VideoDPO/checkpoints/vc2/ref_model.ckpt \
  /mnt/nas/hj/H20_Video_inpainting_DPO/external/VideoDPO/checkpoints/vc2/ref_model.ckpt

rsync -ahP --partial --append-verify -e "$SSH" \
  "$H20":/home/nvme01/H20_Video_inpainting_DPO/logs/videodpo_vc2_dpo/h20-vc2-dpo-official-full-gpu0-7_resume-step2495_20260514_195149/checkpoints/last.ckpt \
  /mnt/nas/hj/h20_vbench_assets/checkpoints/vc2_dpo_last.ckpt
```

#### 18.5.2 VideoDPO conda env 迁移

H20 上 `videodpo` env 完整依赖检查：

```text
/home/nvme01/conda_envs/videodpo:
torch OK
pytorch_lightning OK
omegaconf OK
einops OK
decord OK
av OK
imageio OK
cv2 OK
transformers OK
open_clip OK
kornia OK
fairscale OK
peft OK
```

H20 打包：

```bash
/home/nvme01/conda_envs/videodpo/bin/python -m pip install --user conda-pack
/home/ubuntu/.local/bin/conda-pack \
  -p /home/nvme01/conda_envs/videodpo \
  -o /home/nvme01/H20_Video_inpainting_DPO/data/hf_upload/videodpo_env_for_pai.tar.gz

cd /home/nvme01/H20_Video_inpainting_DPO/data/hf_upload
sha256sum videodpo_env_for_pai.tar.gz > videodpo_env_for_pai.tar.gz.sha256
```

产物：

```text
/home/nvme01/H20_Video_inpainting_DPO/data/hf_upload/videodpo_env_for_pai.tar.gz
  size: 2.8G / transferred 2.98G

/home/nvme01/H20_Video_inpainting_DPO/data/hf_upload/videodpo_env_for_pai.tar.gz.sha256
```

PAI 拉取并解包：

```bash
mkdir -p /mnt/nas/hj/env_packs
rsync -ahP --partial --append-verify \
  -e "ssh -p 22 -o ServerAliveInterval=30 -o ServerAliveCountMax=120 -o ConnectTimeout=30" \
  ubuntu@27.190.15.128:'/home/nvme01/H20_Video_inpainting_DPO/data/hf_upload/videodpo_env_for_pai.tar.gz*' \
  /mnt/nas/hj/env_packs/

cd /mnt/nas/hj/env_packs
sha256sum -c videodpo_env_for_pai.tar.gz.sha256

rm -rf /mnt/nas/hj/conda_envs/videodpo
mkdir -p /mnt/nas/hj/conda_envs/videodpo
tar -xzf /mnt/nas/hj/env_packs/videodpo_env_for_pai.tar.gz \
  -C /mnt/nas/hj/conda_envs/videodpo
/mnt/nas/hj/conda_envs/videodpo/bin/conda-unpack
```

PAI env 验证：

```text
python = /mnt/nas/hj/conda_envs/videodpo/bin/python
torch OK
pytorch_lightning OK
omegaconf OK
einops OK
decord OK
av OK
imageio OK
cv2 OK
transformers OK
open_clip OK
kornia OK
fairscale OK
peft OK
```

`setproctitle` 最初在 `/mnt/nas/hj/conda_envs/videodpo` 缺失，后来安装成功：

```bash
/mnt/nas/hj/conda_envs/videodpo/bin/pip install setproctitle
```

安装后：

```text
setproctitle-1.3.7 installed
```

这使得新启动的任务更有机会在 `nvidia-smi` 中显示 `lingbot-world-model`。旧任务如果在安装前启动，仍可能显示为 python。

### 18.6 VideoDPO VBench smoke 修复过程

#### 18.6.1 第一次失败：拿错环境

H20 上第一次 smoke 时，脚本实际使用了缺少依赖的环境，报错：

```text
ModuleNotFoundError: No module named 'pytorch_lightning'
```

修复：wrapper 强制使用 `CONDA_ENV` 下的 python，并打印：

```text
[videodpo-vbench] python=/home/nvme01/conda_envs/videodpo/bin/python
[videodpo-vbench] sys.executable=/home/nvme01/conda_envs/videodpo/bin/python
```

#### 18.6.2 第二次失败：savefps 是 str

VideoDPO 原始 `scripts/inference.py` 中：

```text
--savefps type=str
```

在 `torchvision.io.write_video` / PyAV 中触发：

```text
AttributeError: 'str' object has no attribute 'numerator'
```

修复：runtime patch 为：

```text
--savefps type=int
```

#### 18.6.3 第三次失败：PyAV pict_type 类型不兼容

继续触发：

```text
TypeError: an integer is required
```

发生在：

```text
torchvision.io.video.py
frame.pict_type = "NONE"
```

修复：runtime patch `inference_utils.py` 的 `save_videos`，优先用 `imageio.mimsave`，失败再用 cv2 writer。

#### 18.6.4 H20 smoke 成功

H20 成功的 ordinary VC2/DPO smoke：

```text
/home/nvme01/H20_Video_inpainting_DPO/logs/vbench_smoke/vc2_vs_dpo_20260516_115117
```

结果：

```text
vc2_base: 2 prompts generated, vbench_standard_named written=2 missing=0
vc2_dpo:  2 prompts generated, vbench_standard_named written=2 missing=0
```

#### 18.6.5 PAI smoke 成功

PAI 成功的 ordinary VC2/DPO smoke：

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_smoke/pai_vc2_vs_dpo_20260517_013718
```

结果：

```text
vc2_base: 2 prompts generated, vbench_standard_named written=2 missing=0
vc2_dpo:  2 prompts generated, vbench_standard_named written=2 missing=0
```

该 smoke 只验证 PAI env、weights、VideoDPO inference、writer、VBench naming helper 可用；不是定量评测。

### 18.7 PAI fullmask-DiffuEraser-DPO 5 epoch 训练完成

PAI 上 fullmask-DiffuEraser-DPO formal run 已完成：

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260516_024051_lingbot-world-model-fullmask-videodpo-epoch5-gpu4-7-20260516_104046
```

训练关键配置：

```text
Dataset Type: videodpo_fullmask
DPO Data Root: /mnt/nas/hj/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.pai.yaml
Ref Model: /mnt/nas/hj/weights/diffuEraser/converted_weights_step48000
GPUs: 4
Train Size: 320x512
Max Epochs: 5
Max Steps: None
Logging Steps: 499
Beta DPO: 5000.0
SFT Reg Weight: 0.0
Lose Gap Weight: 1.0
LR: 6e-06
gradient_accumulation_steps: 4
Mixed Precision: no
VAE dtype: fp32
Policy dtype: auto
Ref dtype: auto
Text dtype: auto
Grad Ckpt: True
Split Pos/Neg: False
```

训练进度终点：

```text
Epoch 5/5: 3125/3125
elapsed: about 16:04:18
```

保存产物：

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260516_024051_lingbot-world-model-fullmask-videodpo-epoch5-gpu4-7-20260516_104046/last_weights/unet_main/config.json
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260516_024051_lingbot-world-model-fullmask-videodpo-epoch5-gpu4-7-20260516_104046/last_weights/unet_main/diffusion_pytorch_model.safetensors
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260516_024051_lingbot-world-model-fullmask-videodpo-epoch5-gpu4-7-20260516_104046/last_weights/brushnet/config.json
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260516_024051_lingbot-world-model-fullmask-videodpo-epoch5-gpu4-7-20260516_104046/last_weights/brushnet/diffusion_pytorch_model.safetensors
```

该 `last_weights` 是 fullmask-DiffuEraser DPO 评测中的 DPO model。

### 18.8 定性 vs 定量：必须严格区分

定性 smoke：

```text
PROMPT_RANDOM_LIMIT=30
SAMPLES_PER_PROMPT=1
RUN_VBENCH=0
MAKE_QUAL_COMPARE=1
```

用途：

```text
只看真实生成视频效果和 base/DPO side-by-side。
不参与论文式 VBench 数值。
```

定量 full VBench：

```text
PROMPTS_FILE=external/VideoDPO/prompts/vbench_standard_prompts.txt
PROMPT_LIMIT=0
PROMPT_RANDOM_LIMIT must not be set
SAMPLES_PER_PROMPT=5
RUN_VBENCH=1
```

用途：

```text
生成完整 VBench standard prompt suite。
使用 VBench evaluate.py 计算维度分数。
汇总为 paper-style table。
```

这是本轮最终结论之一：VBench 定量不能使用随机 30 条，也不能使用 `PROMPT_LIMIT=2` smoke 输出。

### 18.9 当前评测分工与 GPU 分配

最后确定的分卡原则：

```text
终端 1：ordinary VideoDPO VC2-base vs VC2-DPO，使用 GPU 0,1,2,3
终端 2：fullmask DiffuEraser-base vs FullMask-DPO，使用 GPU 4,5,6,7
```

但需要注意：

```text
h20_diffueraser_fullmask_vbench.sh 当前每个模型是单卡 generation/eval。
因此 fullmask baseline 和 DPO 同时跑时，实际通常只占 GPU 4 和 GPU 5。
GPU 6/7 可留作后续 VBench eval 或其他扩展；若想 fullmask generation 也像 VideoDPO 一样切 prompt 多卡，需要再扩展 fullmask generator。
```

曾经启动过的中间状态：

```text
ordinary VC2/DPO full run 曾以 CUDA_VISIBLE_DEVICES=2,3,4,5 启动：
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_full/pai_vc2_vs_dpo_full_20260517_033604

fullmask baseline 曾以 CUDA_VISIBLE_DEVICES=6 单独启动：
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_033619/diffueraser_base
```

随后分卡方案被修正为：

```text
ordinary: 0,1,2,3
fullmask: 4,5,6,7
```

实际运行状态必须以 PAI 第三个终端 audit 为准，不应仅凭上述中间日志判断。

### 18.10 ordinary VideoDPO full VBench 标准命令

终端 1 使用。该命令会完整运行：

```text
vc2_base generation
vc2_base VBench eval
vc2_dpo generation
vc2_dpo VBench eval
paper-style table export
```

命令：

```bash
cd /mnt/nas/hj/H20_Video_inpainting_DPO

jobs -l
kill -TERM %1 2>/dev/null || true
pkill -TERM -f 'logs/vbench_full/pai_vc2_vs_dpo_full_' 2>/dev/null || true
sleep 10

VIDEODPO_ENV=/mnt/nas/hj/conda_envs/videodpo
BASE_CKPT=$PWD/external/VideoDPO/checkpoints/vc2/model.ckpt
DPO_CKPT=/mnt/nas/hj/h20_vbench_assets/checkpoints/vc2_dpo_last.ckpt
OUT_ROOT=$PWD/logs/vbench_full/pai_vc2_vs_dpo_full_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT_ROOT"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WORLDMODELPHY_PROCESS_NAME=lingbot-world-model \
PROCESS_TITLE=lingbot-world-model \
CONDA_ENV="$VIDEODPO_ENV" \
PROJECT_ROOT=$PWD \
PROJECT_HOME=/mnt/nas/hj \
PROJECT_DEV=/mnt/nas/hj \
PROJECT_DATA=$PWD \
CKPT_SPECS="vc2_base:$BASE_CKPT,vc2_dpo:$DPO_CKPT" \
OUT_ROOT="$OUT_ROOT" \
NUM_INFER_GPUS=4 \
SAMPLES_PER_PROMPT=5 \
RUN_INFERENCE=1 \
RUN_VBENCH=1 \
MAKE_QUAL_COMPARE=0 \
MAKE_PAPER_TABLE=1 \
bash DPO_finetune/scripts/sc_videodpo_vc2_vbench.sbatch 2>&1 | tee "$OUT_ROOT/full_vbench.log"
```

输出表格：

```text
$OUT_ROOT/vbench_paper_table.md
$OUT_ROOT/vbench_paper_table.csv
$OUT_ROOT/vbench_paper_table.tex
```

### 18.11 fullmask DiffuEraser full VBench 标准命令

终端 2 使用。该命令会完整运行：

```text
DiffuEraser baseline fullmask generation/eval
FullMask-DPO generation/eval
paper-style table export
```

命令：

```bash
cd /mnt/nas/hj/H20_Video_inpainting_DPO

jobs -l
kill -TERM %1 %2 2>/dev/null || true
pkill -TERM -f 'logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_' 2>/dev/null || true
sleep 10

PAIR_ROOT=$PWD/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_$(date +%Y%m%d_%H%M%S)
PROMPTS=$PWD/external/VideoDPO/prompts/vbench_standard_prompts.txt
BASE=/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000
FM_DPO=/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260516_024051_lingbot-world-model-fullmask-videodpo-epoch5-gpu4-7-20260516_104046/last_weights
mkdir -p "$PAIR_ROOT"

CUDA_VISIBLE_DEVICES=4 \
WORLDMODELPHY_PROCESS_NAME=lingbot-world-model \
PROCESS_TITLE=lingbot-world-model \
CONDA_ENV=/mnt/nas/hj/conda_envs/diffueraser \
VBENCH_CONDA_ENV=/mnt/nas/hj/conda_envs/videodpo \
PROJECT_ROOT=$PWD \
WEIGHTS_DIR=/mnt/nas/hj/weights \
WEIGHTS_PATH="$BASE" \
OUT_ROOT="$PAIR_ROOT/diffueraser_base" \
PROMPTS_FILE="$PROMPTS" \
SAMPLES_PER_PROMPT=5 \
PROMPT_LIMIT=0 \
RUN_VBENCH=1 \
bash DPO_finetune/scripts/h20_diffueraser_fullmask_vbench.sh 2>&1 | tee "$PAIR_ROOT/diffueraser_base.log" &

CUDA_VISIBLE_DEVICES=5 \
WORLDMODELPHY_PROCESS_NAME=lingbot-world-model \
PROCESS_TITLE=lingbot-world-model \
CONDA_ENV=/mnt/nas/hj/conda_envs/diffueraser \
VBENCH_CONDA_ENV=/mnt/nas/hj/conda_envs/videodpo \
PROJECT_ROOT=$PWD \
WEIGHTS_DIR=/mnt/nas/hj/weights \
WEIGHTS_PATH="$FM_DPO" \
OUT_ROOT="$PAIR_ROOT/diffueraser_dpo" \
PROMPTS_FILE="$PROMPTS" \
SAMPLES_PER_PROMPT=5 \
PROMPT_LIMIT=0 \
RUN_VBENCH=1 \
bash DPO_finetune/scripts/h20_diffueraser_fullmask_vbench.sh 2>&1 | tee "$PAIR_ROOT/diffueraser_dpo.log" &

wait

/mnt/nas/hj/conda_envs/videodpo/bin/python tools/vbench_make_paper_table.py \
  --out_root "$PAIR_ROOT" \
  --ckpt_specs "diffueraser_base:$BASE,diffueraser_dpo:$FM_DPO" \
  --model_group "DiffuEraser-FullMask" \
  --label_map "diffueraser_base=DiffuEraser,diffueraser_dpo=FullMask-DPO" \
  --output_prefix "$PAIR_ROOT/vbench_paper_table"

echo "DONE fullmask eval:"
echo "$PAIR_ROOT"
echo "$PAIR_ROOT/vbench_paper_table.md"
```

输出表格：

```text
$PAIR_ROOT/vbench_paper_table.md
$PAIR_ROOT/vbench_paper_table.csv
$PAIR_ROOT/vbench_paper_table.tex
```

### 18.12 ordinary VideoDPO 30 prompt 定性 smoke 命令

该命令只用于定性，不用于定量：

```bash
cd /mnt/nas/hj/H20_Video_inpainting_DPO

VIDEODPO_ENV=/mnt/nas/hj/conda_envs/videodpo
BASE_CKPT=$PWD/external/VideoDPO/checkpoints/vc2/model.ckpt
DPO_CKPT=/mnt/nas/hj/h20_vbench_assets/checkpoints/vc2_dpo_last.ckpt
OUT_ROOT=$PWD/logs/vbench_qual_smoke/pai_vc2_vs_dpo_random30_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT_ROOT"

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
WORLDMODELPHY_PROCESS_NAME=lingbot-world-model \
PROCESS_TITLE=lingbot-world-model \
CONDA_ENV="$VIDEODPO_ENV" \
PROJECT_ROOT=$PWD \
PROJECT_HOME=/mnt/nas/hj \
PROJECT_DEV=/mnt/nas/hj \
PROJECT_DATA=$PWD \
CKPT_SPECS="vc2_base:$BASE_CKPT,vc2_dpo:$DPO_CKPT" \
OUT_ROOT="$OUT_ROOT" \
NUM_INFER_GPUS=6 \
PROMPT_RANDOM_LIMIT=30 \
PROMPT_RANDOM_SEED=20260517 \
SAMPLES_PER_PROMPT=1 \
RUN_INFERENCE=1 \
RUN_VBENCH=0 \
MAKE_QUAL_COMPARE=1 \
bash DPO_finetune/scripts/sc_videodpo_vc2_vbench.sbatch 2>&1 | tee "$OUT_ROOT.smoke.log"

find "$OUT_ROOT/qual_side_by_side" -name '*.mp4' | wc -l
```

注意：`tee` 需要 `$OUT_ROOT` 预先存在，否则会出现：

```text
tee: ... No such file or directory
```

### 18.13 fullmask DiffuEraser 30 prompt 定性 smoke 命令

该命令只用于定性，不用于定量：

```bash
cd /mnt/nas/hj/H20_Video_inpainting_DPO

PAIR_ROOT=$PWD/logs/fullmask_qual_smoke/pai_fullmask_random30_$(date +%Y%m%d_%H%M%S)
PROMPTS=$PAIR_ROOT/prompts_random_30.txt
BASE=/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000
FM_DPO=/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260516_024051_lingbot-world-model-fullmask-videodpo-epoch5-gpu4-7-20260516_104046/last_weights
mkdir -p "$PAIR_ROOT"

/mnt/nas/hj/conda_envs/videodpo/bin/python tools/sample_prompts.py \
  external/VideoDPO/prompts/vbench_standard_prompts.txt \
  "$PROMPTS" 30 20260517

CUDA_VISIBLE_DEVICES=4 \
WORLDMODELPHY_PROCESS_NAME=lingbot-world-model \
PROCESS_TITLE=lingbot-world-model \
CONDA_ENV=/mnt/nas/hj/conda_envs/diffueraser \
PROJECT_ROOT=$PWD \
WEIGHTS_DIR=/mnt/nas/hj/weights \
WEIGHTS_PATH="$BASE" \
OUT_ROOT="$PAIR_ROOT/diffueraser_base" \
PROMPTS_FILE="$PROMPTS" \
SAMPLES_PER_PROMPT=1 \
PROMPT_LIMIT=0 \
RUN_VBENCH=0 \
bash DPO_finetune/scripts/h20_diffueraser_fullmask_vbench.sh 2>&1 | tee "$PAIR_ROOT/diffueraser_base.log" &

CUDA_VISIBLE_DEVICES=5 \
WORLDMODELPHY_PROCESS_NAME=lingbot-world-model \
PROCESS_TITLE=lingbot-world-model \
CONDA_ENV=/mnt/nas/hj/conda_envs/diffueraser \
PROJECT_ROOT=$PWD \
WEIGHTS_DIR=/mnt/nas/hj/weights \
WEIGHTS_PATH="$FM_DPO" \
OUT_ROOT="$PAIR_ROOT/diffueraser_dpo" \
PROMPTS_FILE="$PROMPTS" \
SAMPLES_PER_PROMPT=1 \
PROMPT_LIMIT=0 \
RUN_VBENCH=0 \
bash DPO_finetune/scripts/h20_diffueraser_fullmask_vbench.sh 2>&1 | tee "$PAIR_ROOT/diffueraser_dpo.log" &

wait

/mnt/nas/hj/conda_envs/videodpo/bin/python tools/vbench_named_side_by_side.py \
  --left_dir "$PAIR_ROOT/diffueraser_base/vbench_standard_named" \
  --right_dir "$PAIR_ROOT/diffueraser_dpo/vbench_standard_named" \
  --prompts_file "$PROMPTS" \
  --output_dir "$PAIR_ROOT/qual_side_by_side" \
  --left_label "DiffuEraser" \
  --right_label "FullMask-DPO" \
  --sample_index 0 \
  --fps 10 \
  --strict

find "$PAIR_ROOT/qual_side_by_side" -name '*.mp4' | wc -l
```

如果 PAI 还没有 `9160ab4`，`tools/sample_prompts.py` 会不存在。这不影响 full VBench 定量；只影响 30 prompt 定性 smoke。可直接 `git pull` 到 `9160ab4` 或从 H20 同步该文件。

### 18.14 第三个终端 runtime audit 命令

由于这里的本地工具运行在 `/home/hj` 机器，不是 PAI，无法直接代替 PAI 新开终端。PAI 第三个终端应使用以下审查脚本，只读检查，不杀任务、不改文件：

```bash
cd /mnt/nas/hj/H20_Video_inpainting_DPO || exit 1

AUDIT_LOG=/tmp/pai_vbench_runtime_audit_$(date +%Y%m%d_%H%M%S).log

bash -s <<'BASH' 2>&1 | tee "$AUDIT_LOG"
set -u

ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO

echo "========== Context =========="
hostname
date
cd "$ROOT" && git rev-parse --short HEAD

echo
echo "========== GPU =========="
nvidia-smi

echo
echo "========== Running Processes =========="
ps -eo pid,ppid,pgid,etime,stat,pcpu,pmem,args \
  | egrep 'lingbot-world-model|sc_videodpo_vc2_vbench|h20_diffueraser_fullmask_vbench|scripts/inference.py|generate_diffueraser_fullmask_vbench|evaluate.py' \
  | grep -v egrep || true

echo
echo "========== Process Envs =========="
for PID in $(pgrep -f 'lingbot-world-model|sc_videodpo_vc2_vbench|h20_diffueraser_fullmask_vbench|scripts/inference.py|generate_diffueraser_fullmask_vbench|evaluate.py' | sort -u); do
  echo "----- PID=$PID -----"
  ps -fp "$PID" || true
  echo "cwd=$(readlink -f /proc/$PID/cwd 2>/dev/null || true)"
  tr '\0' '\n' < /proc/$PID/environ 2>/dev/null \
    | egrep 'CUDA_VISIBLE_DEVICES|WORLDMODELPHY|PROCESS_TITLE|OUT_ROOT|PAIR_ROOT|CKPT_SPECS|WEIGHTS_PATH|CONDA_ENV|VBENCH_CONDA_ENV|PROMPTS_FILE|SAMPLES_PER_PROMPT|RUN_VBENCH|NUM_INFER_GPUS' || true
done

echo
echo "========== Latest Roots =========="
VC2_ROOT=$(ls -td "$ROOT"/logs/vbench_full/pai_vc2_vs_dpo_full_* 2>/dev/null | head -1 || true)
FM_ROOT=$(ls -td "$ROOT"/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_* 2>/dev/null | head -1 || true)
echo "VC2_ROOT=$VC2_ROOT"
echo "FM_ROOT=$FM_ROOT"

check_root () {
  local name="$1"
  local root="$2"
  echo
  echo "========== $name =========="
  if [[ -z "$root" || ! -d "$root" ]]; then
    echo "[WARN] root missing: $root"
    return
  fi

  echo "[INFO] root=$root"
  echo "[INFO] generated mp4 counts:"
  find "$root" -type f -name '*.mp4' | wc -l
  find "$root" -path '*vbench_standard_named*' -type f -name '*.mp4' | wc -l | sed 's/^/[INFO] vbench_named_mp4=/'

  echo
  echo "[INFO] summaries/tables:"
  find "$root" -type f \( -name 'summary.json' -o -name 'summary.csv' -o -name 'vbench_paper_table.md' -o -name 'vbench_paper_table.csv' -o -name 'vbench_paper_table.tex' \) -print

  echo
  echo "[INFO] tail main logs:"
  for f in "$root"/full_vbench.log "$root"/diffueraser_base.log "$root"/diffueraser_dpo.log; do
    [[ -f "$f" ]] && { echo "----- tail $f -----"; tail -40 "$f"; }
  done

  echo
  echo "[CHECK] errors:"
  ERR=$(find "$root" -type f \( -name '*.log' -o -name '*.txt' \) -print0 2>/dev/null \
    | xargs -0 grep -nEi 'Traceback|RuntimeError|CUDA out of memory|out of memory|No such file|ModuleNotFoundError|FileNotFoundError|Killed|failed|error' 2>/dev/null \
    | grep -viE 'FutureWarning|UserWarning|warning|error=0|missing=0|savefps type already patched|video writer already patched|multi-gpu env ranks already patched' \
    | tail -80 || true)
  if [[ -n "$ERR" ]]; then
    echo "$ERR"
    echo "[BAD] $name has suspicious error lines above"
  else
    echo "[OK] no suspicious error lines found in $name logs"
  fi
}

check_root "VC2 / DPO full VBench" "$VC2_ROOT"
check_root "FullMask DiffuEraser full VBench" "$FM_ROOT"

echo
echo "审查日志: $AUDIT_LOG"
BASH
```

### 18.15 nvidia-smi 进程名与旧任务说明

用户要求：

```text
不管谁 nvidia-smi，都显示 lingbot-world-model。
```

当前实现依赖：

```text
WORLDMODELPHY_PROCESS_NAME=lingbot-world-model
PROCESS_TITLE=lingbot-world-model
sitecustomize.py / setproctitle
```

注意：

```text
1. setproctitle 安装前启动的旧进程可能仍显示 .../bin/python。
2. Linux /proc/comm 有 15 字符截断限制。
3. nvidia-smi 的 Process name 在不同驱动/容器上可能显示 argv、python path 或 setproctitle 后的 title。
4. 旧 GPU0 PID=1789284 经检查属于我们自己的旧 VideoDPO VBench full run，不是别人的任务。
```

旧 GPU0 任务环境曾显示：

```text
WORLDMODELPHY_PROCESS_NAME=lingbot-world-model
PROCESS_TITLE=lingbot-world-model
CONDA_ENV=/mnt/nas/hj/conda_envs/videodpo
SAMPLES_PER_PROMPT=5
CKPT_SPECS=vc2_base:...,vc2_dpo:...
OUT_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_full/pai_vc2_vs_dpo_20260517_030752
PROMPT_LIMIT=0
cwd=/mnt/nas/hj/H20_Video_inpainting_DPO/external/VideoDPO
```

它是在修复 setproctitle 前启动的，所以 `nvidia-smi` 中可能显示 python。

### 18.16 当前仍需完成/确认

1. PAI 是否已经拉到 `9160ab4`：

```bash
cd /mnt/nas/hj/H20_Video_inpainting_DPO
git rev-parse --short HEAD
```

如果还是 `65eefe0`，定量 VBench 不受影响；若要用 `tools/sample_prompts.py` 做定性 smoke，需要拉到 `9160ab4` 或同步该文件。

2. 确认终端 1 / 终端 2 实际分卡：

```text
terminal 1 ordinary VC2/DPO: GPU 0,1,2,3
terminal 2 fullmask base/DPO: GPU 4,5; GPU 6/7 可能空闲
```

3. VBench full 定量完成后收集：

ordinary VideoDPO：

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_full/<run>/vbench_paper_table.md
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_full/<run>/vbench_paper_table.csv
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_full/<run>/vbench_paper_table.tex
```

fullmask DiffuEraser：

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/<run>/vbench_paper_table.md
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/<run>/vbench_paper_table.csv
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/<run>/vbench_paper_table.tex
```

4. 定性 side-by-side 完成后收集：

ordinary VideoDPO：

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_qual_smoke/<run>/qual_side_by_side/*.mp4
```

fullmask DiffuEraser：

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/fullmask_qual_smoke/<run>/qual_side_by_side/*.mp4
```

5. paper 对齐说明：

```text
当前 paper-style table 只使用 VBench summary 中真实存在的指标。
HPS (V) / PickScore 目前没有在该 wrapper 中计算，必须标 "-" 或后续单独跑对应 evaluator。
不能把 VBench 分数伪装成 HPS/PickScore。
```

6. fullmask 速度与多 GPU：

```text
ordinary VideoDPO wrapper 已支持 prompt-level 多 GPU inference。
fullmask DiffuEraser wrapper 当前是每模型单卡。
如果 fullmask full VBench 太慢，下一步应扩展 generate_diffueraser_fullmask_vbench.py 支持 rank/world_size prompt slicing，并在 h20_diffueraser_fullmask_vbench.sh 中加 NUM_INFER_GPUS。
```

## 19. 2026-05-17 续写：长跑审查、整体框架、代码构思与高风险坑位

本节继续只追加，不覆盖前面内容。它记录截至 2026-05-17 CST 上午约 11:24 的真实运行状态，并把本轮形成的代码框架、评测框架、排错逻辑、已知坑位整理成接手文档。

### 19.1 截至 2026-05-17 11:24 CST 的真实运行状态

PAI 最新 runtime audit 显示：

```text
hostname: dsw-753014-dc85766cb-4v2jj
time: Sun May 17 11:24:06 AM CST 2026
git commit on PAI: be96042
```

当前两个正式 full VBench 任务都还在跑，没有扫到真实错误：

```text
VC2 / DPO full VBench: OK
FullMask DiffuEraser full VBench: OK
```

对应 root：

```text
ordinary VideoDPO root:
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_full/pai_vc2_vs_dpo_full_20260517_034228

fullmask DiffuEraser root:
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_035156
```

GPU 分布：

```text
GPU 0,1,2,3:
ordinary VideoDPO VC2-base / VC2-DPO wrapper
current process path: /mnt/nas/hj/conda_envs/videodpo/bin/python scripts/inference.py
CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_INFER_GPUS=4

GPU 4:
fullmask DiffuEraser baseline
process name: lingbot-world-model
CUDA_VISIBLE_DEVICES=4
WEIGHTS_PATH=/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000

GPU 5:
fullmask DiffuEraser DPO
process name: lingbot-world-model
CUDA_VISIBLE_DEVICES=5
WEIGHTS_PATH=/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260516_024051_lingbot-world-model-fullmask-videodpo-epoch5-gpu4-7-20260516_104046/last_weights

GPU 6,7:
idle at audit time
reason: current fullmask wrapper is one model per one GPU and has not implemented prompt-level multi-GPU sharding.
```

ordinary VideoDPO progress at audit:

```text
vc2_base raw mp4: 4176 / 4730
vc2_dpo raw mp4: 0 / 4730
current phase: vc2_base sample_4, seed=20230215
completed prepare-vbench for sample_0, sample_1, sample_2, sample_3:
  written=946 missing=0 for each completed sample
current sample_4 is still generating raw mp4
```

ordinary VideoDPO command evidence:

```text
CKPT_SPECS=vc2_base:/mnt/nas/hj/H20_Video_inpainting_DPO/external/VideoDPO/checkpoints/vc2/model.ckpt,vc2_dpo:/mnt/nas/hj/h20_vbench_assets/checkpoints/vc2_dpo_last.ckpt
SAMPLES_PER_PROMPT=5
RUN_VBENCH=1
PROMPTS_FILE=/mnt/nas/hj/H20_Video_inpainting_DPO/external/VideoDPO/prompts/vbench_standard_prompts.txt
```

fullmask DiffuEraser progress at audit:

```text
diffueraser_base mp4: 2512 / 4730
diffueraser_dpo mp4: 2513 / 4730
current phase: sample=2, prompt around 600/946
SAMPLES_PER_PROMPT=5
RUN_VBENCH=1
PROMPTS_FILE=/mnt/nas/hj/H20_Video_inpainting_DPO/external/VideoDPO/prompts/vbench_standard_prompts.txt
```

Interpretation：

```text
ordinary VideoDPO 还在 baseline generation 阶段，尚未进入 vc2_dpo generation。
fullmask baseline 和 fullmask DPO 正在并行 generation，数量基本同步。
两个任务都还没有进入最终 VBench evaluate/table 阶段，因此当前还没有 summary.json / vbench_paper_table.md。
```

### 19.2 当前评测系统的整体框架

整个评测系统被拆成两个互相对齐但实现不同的 pipeline。

#### 19.2.1 ordinary VideoDPO VC2/DPO pipeline

目标：

```text
复现 VideoDPO paper 中 VC2 baseline 与 VC2-DPO 的 VBench comparison。
```

输入：

```text
prompt suite:
/mnt/nas/hj/H20_Video_inpainting_DPO/external/VideoDPO/prompts/vbench_standard_prompts.txt

baseline checkpoint:
/mnt/nas/hj/H20_Video_inpainting_DPO/external/VideoDPO/checkpoints/vc2/model.ckpt

DPO checkpoint:
/mnt/nas/hj/h20_vbench_assets/checkpoints/vc2_dpo_last.ckpt

env:
/mnt/nas/hj/conda_envs/videodpo
```

核心 wrapper：

```text
DPO_finetune/scripts/sc_videodpo_vc2_vbench.sbatch
```

执行框架：

```text
for label in [vc2_base, vc2_dpo]:
  for sample_idx in [0,1,2,3,4]:
    seed = 20230211 + sample_idx
    launch N ranks of VideoDPO scripts/inference.py
    each rank handles a slice of VBench prompts
    raw outputs are 0001.mp4, 0002.mp4, ...
    after all ranks finish:
      tools/videodpo_prepare_vbench_standard.py
      maps raw index names to VBench standard names:
        <prompt>-<sample_idx>.mp4

  if RUN_VBENCH=1:
    VBench evaluate.py over vbench_standard_named
    tools/summarize_vbench_results.py -> summary.json/summary.csv

after all labels finish:
  tools/vbench_make_paper_table.py -> vbench_paper_table.md/csv/tex
```

输出目录结构：

```text
$OUT_ROOT/
  full_vbench.log
  vc2_base/
    raw/
      sample_0/
        0001.mp4
        infer_rank0.log
        ...
      sample_1/
      ...
    vbench_standard_named/
      <prompt>-0.mp4
      <prompt>-1.mp4
      ...
    vbench_eval/
      results_*_eval_results.json
      summary.json
      summary.csv
  vc2_dpo/
    raw/
    vbench_standard_named/
    vbench_eval/
  vbench_paper_table.md
  vbench_paper_table.csv
  vbench_paper_table.tex
```

#### 19.2.2 fullmask DiffuEraser baseline/DPO pipeline

目标：

```text
在和 ordinary VideoDPO 相同的 VBench prompt suite 上，对比：
  original DiffuEraser fullmask baseline
  fullmask-DiffuEraser-DPO finetuned model
```

输入：

```text
prompt suite:
/mnt/nas/hj/H20_Video_inpainting_DPO/external/VideoDPO/prompts/vbench_standard_prompts.txt

baseline:
/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000

DPO:
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260516_024051_lingbot-world-model-fullmask-videodpo-epoch5-gpu4-7-20260516_104046/last_weights

generation env:
/mnt/nas/hj/conda_envs/diffueraser

VBench env:
/mnt/nas/hj/conda_envs/videodpo
```

核心 wrapper：

```text
DPO_finetune/scripts/h20_diffueraser_fullmask_vbench.sh
```

生成器：

```text
tools/generate_diffueraser_fullmask_vbench.py
```

执行框架：

```text
run baseline and DPO in parallel:
  CUDA_VISIBLE_DEVICES=4 -> baseline
  CUDA_VISIBLE_DEVICES=5 -> DPO

for each model:
  build DiffuEraser StageOne pipeline
  load SD1.5 / VAE / text_encoder / tokenizer
  load unet_main + brushnet from WEIGHTS_PATH
  if unet_main is UNetMotionModel:
    extract 2D UNet weights for StageOne pipeline
  create blank RGB frames
  create full masks
  for sample_idx in [0,1,2,3,4]:
    seed = 20230211 + sample_idx
    for prompt in 946 VBench prompts:
      run fullmask generation
      write <prompt>-<sample_idx>.mp4

  if RUN_VBENCH=1:
    run VBench evaluate.py over vbench_standard_named
    write summary.json/summary.csv

after baseline and DPO both finish:
  tools/vbench_make_paper_table.py over parent PAIR_ROOT
```

输出目录结构：

```text
$PAIR_ROOT/
  diffueraser_base.log
  diffueraser_dpo.log
  diffueraser_base/
    vbench_standard_named/
      <prompt>-0.mp4
      <prompt>-1.mp4
      ...
    vbench_eval/
      summary.json
      summary.csv
  diffueraser_dpo/
    vbench_standard_named/
    vbench_eval/
  vbench_paper_table.md
  vbench_paper_table.csv
  vbench_paper_table.tex
```

### 19.3 代码构思：为什么这么改

#### 19.3.1 为什么 VideoDPO wrapper 需要显式 Python

最早 H20 smoke 失败的根因是 wrapper 拿错了 Python。`phys-main` 环境缺：

```text
pytorch_lightning
av
open_clip
kornia
fairscale
```

而 VideoDPO inference 需要这些依赖。于是 `sc_videodpo_vc2_vbench.sbatch` 必须显式从：

```text
CONDA_ENV=/mnt/nas/hj/conda_envs/videodpo
PYTHON_BIN=$CONDA_ENV/bin/python
```

调用：

```text
$PYTHON_BIN scripts/inference.py
```

并打印：

```text
[videodpo-vbench] python=...
[videodpo-vbench] sys.executable=...
```

这个打印是必要审计点，能立即确认没有拿错 env。

#### 19.3.2 为什么 VideoDPO inference 需要 runtime patch

VideoDPO upstream `scripts/inference.py` 默认只单进程：

```python
rank, gpu_num = 0, 1
run_inference(args, gpu_num, rank)
```

为了不大改 external repo，又要支持 prompt-level 多 GPU，wrapper 在运行前 patch main block 为：

```python
rank = int(os.environ.get("VIDEODPO_INFER_RANK", "0"))
gpu_num = int(os.environ.get("VIDEODPO_INFER_NUM_GPUS", "1"))
run_inference(args, gpu_num, rank)
```

这样 wrapper 可以在同一个 sample 下启动多个 rank：

```text
VIDEODPO_INFER_RANK=0 VIDEODPO_INFER_NUM_GPUS=4 python scripts/inference.py ...
VIDEODPO_INFER_RANK=1 VIDEODPO_INFER_NUM_GPUS=4 python scripts/inference.py ...
VIDEODPO_INFER_RANK=2 VIDEODPO_INFER_NUM_GPUS=4 python scripts/inference.py ...
VIDEODPO_INFER_RANK=3 VIDEODPO_INFER_NUM_GPUS=4 python scripts/inference.py ...
```

VideoDPO 自己会按 `rank/gpu_num` 切分 prompt。这样 GPU 0-3 可以并行生成同一个 sample 的不同 prompt subset。

#### 19.3.3 为什么要 patch VideoDPO writer

VideoDPO upstream 有两个 PyAV/torchvision 兼容问题：

1. `--savefps` 是 str：

```text
AttributeError: 'str' object has no attribute 'numerator'
```

修复：

```python
parser.add_argument("--savefps", type=int, default=10, ...)
```

2. `torchvision.io.write_video` 与当前 PyAV 不兼容：

```text
TypeError: an integer is required
frame.pict_type = "NONE"
```

修复思路：

```text
优先 imageio.mimsave
失败则 cv2.VideoWriter fallback
```

这类 patch 是 runtime idempotent patch：如果已经 patch 过，会打印：

```text
savefps type already patched
video writer already patched
multi-gpu env ranks already patched
```

这些不是错误，不应被 audit 当成错误。

#### 19.3.4 为什么 fullmask writer 也要 fallback

fullmask generator 最初使用：

```python
imageio.mimsave(path, arrays, fps=fps)
```

在 PAI diffueraser env 中触发：

```text
TypeError: expected bytes, NoneType found
```

堆栈在：

```text
imageio/plugins/pyav.py
self._container.add_stream(codec, fps)
```

根因是 imageio pyav plugin 没拿到 codec。修复 commit：

```text
be96042 Fix fullmask VBench video writer
```

修复逻辑：

```python
try:
    imageio.mimsave(path, arrays, fps=int(fps), codec="libx264", macro_block_size=1)
    return
except Exception as imageio_error:
    import cv2
    writer = cv2.VideoWriter(..., fourcc="mp4v", ...)
    ...
```

这使 fullmask 在 PAI 上能稳定写 `.mp4`。修复后 audit 看到：

```text
base mp4 count increasing
dpo mp4 count increasing
no Traceback
```

#### 19.3.5 为什么 fullmask 现在没有用 GPU 6/7

ordinary VideoDPO wrapper 已支持 prompt-level multi-rank inference；fullmask generator 暂时没有。

fullmask 现在的并行粒度是：

```text
one model process per one GPU
baseline -> GPU4
DPO -> GPU5
```

它没有把单个模型的 946 prompts 再切给多个 GPU。因此：

```text
GPU6/7 idle 是正常现象，不表示任务错了。
```

如果要利用 4-7 四张卡，应扩展：

```text
tools/generate_diffueraser_fullmask_vbench.py:
  add --rank
  add --world_size
  prompts = prompts[rank::world_size]

DPO_finetune/scripts/h20_diffueraser_fullmask_vbench.sh:
  add NUM_INFER_GPUS
  for rank in range(NUM_INFER_GPUS):
    CUDA_VISIBLE_DEVICES mapped rank process
```

但这会改变 output completeness checking 和 VBench eval timing，需要确保所有 rank 完成后再 eval。

### 19.4 评测产物与计数规则

VBench standard prompt count：

```text
946 prompts
```

Samples per prompt：

```text
SAMPLES_PER_PROMPT=5
```

因此每个 model 应生成：

```text
946 * 5 = 4730 videos
```

ordinary VideoDPO：

```text
vc2_base expected raw mp4: 4730
vc2_dpo expected raw mp4: 4730

after prepare-vbench:
vc2_base/vbench_standard_named expected mp4: 4730
vc2_dpo/vbench_standard_named expected mp4: 4730
```

fullmask DiffuEraser：

```text
diffueraser_base/vbench_standard_named expected mp4: 4730
diffueraser_dpo/vbench_standard_named expected mp4: 4730
```

Important detail for ordinary VideoDPO：

```text
raw count grows first.
vbench_standard_named only updates after each sample completes.
Therefore during sample_4 generation:
  raw mp4 may be > named mp4
  named mp4 may show only completed samples
This is expected.
```

Important detail for fullmask：

```text
fullmask generator writes VBench-standard names directly.
Therefore mp4 count in vbench_standard_named grows continuously.
```

### 19.5 paper-style table 设计

`tools/vbench_make_paper_table.py` 的目标不是复刻全部 paper 表格，而是从当前真实 VBench summary 中提取可用指标，并以 paper-friendly 格式输出：

```text
Backbone
Model
VBench Total (%)
VBench Quality (%)
VBench Semantics (%)
HPS (V)
PickScore
```

当前只有 VBench 指标可用，所以：

```text
HPS (V) = "-"
PickScore = "-"
```

这是刻意设计，原因：

```text
HPS 和 PickScore 没有在当前 VBench wrapper 中计算。
不能把 VBench 指标伪装成 HPS/PickScore。
如果需要 paper 中完整对齐，必须后续单独跑 HPS/PickScore evaluator。
```

ordinary VideoDPO 表格默认 label map：

```text
vc2_base -> Baseline[7]
vc2_dpo -> VideoDPO
model_group -> VideoCrafter2
```

fullmask DiffuEraser 表格 label map：

```text
diffueraser_base -> DiffuEraser
diffueraser_dpo -> FullMask-DPO
model_group -> DiffuEraser-FullMask
```

### 19.6 定性 side-by-side 设计

定性分析只用于人眼检查，不参与 VBench 分数。

ordinary VideoDPO side-by-side：

```text
tool: tools/videodpo_make_side_by_side.py
input:
  vc2_base/raw/sample_0/0001.mp4
  vc2_dpo/raw/sample_0/0001.mp4
mapping:
  prompt file line order -> 0001,0002,...
output:
  qual_side_by_side/*.mp4
labels:
  VC2-Base
  VC2-DPO
```

fullmask DiffuEraser side-by-side：

```text
tool: tools/vbench_named_side_by_side.py
input:
  diffueraser_base/vbench_standard_named/<prompt>-0.mp4
  diffueraser_dpo/vbench_standard_named/<prompt>-0.mp4
output:
  qual_side_by_side/*.mp4
labels:
  DiffuEraser
  FullMask-DPO
```

定性 prompt 抽样：

```text
tool: tools/sample_prompts.py
input: external/VideoDPO/prompts/vbench_standard_prompts.txt
count: 30
seed: 20260517
```

再次强调：

```text
PROMPT_RANDOM_LIMIT=30 或 sample_prompts.py 只用于定性。
定量 VBench 必须使用完整 vbench_standard_prompts.txt。
```

### 19.7 runtime audit 的正确判断方法

审查时应看这几个层级。

#### 19.7.1 进程层

ordinary VideoDPO 应看到：

```text
bash DPO_finetune/scripts/sc_videodpo_vc2_vbench.sbatch
python scripts/inference.py ...
CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_INFER_GPUS=4
RUN_VBENCH=1
SAMPLES_PER_PROMPT=5
```

fullmask DiffuEraser 应看到：

```text
bash DPO_finetune/scripts/h20_diffueraser_fullmask_vbench.sh
lingbot-world-model
CUDA_VISIBLE_DEVICES=4
WEIGHTS_PATH=/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000

bash DPO_finetune/scripts/h20_diffueraser_fullmask_vbench.sh
lingbot-world-model
CUDA_VISIBLE_DEVICES=5
WEIGHTS_PATH=/mnt/nas/.../last_weights
```

#### 19.7.2 GPU 层

Expected：

```text
GPU 0-3 high util: ordinary VideoDPO
GPU 4 high util: fullmask baseline
GPU 5 high util: fullmask DPO
GPU 6-7 idle: acceptable under current fullmask implementation
```

普通 VideoDPO 进程名可能仍显示为：

```text
.../conda_envs/videodpo/bin/python
```

原因：

```text
external VideoDPO inference 进程可能没有通过 setproctitle hook 改 title。
```

fullmask 进程已经显示：

```text
lingbot-world-model
```

#### 19.7.3 输出层

ordinary：

```bash
find "$VC2_ROOT/vc2_base/raw" -type f -name '*.mp4' | wc -l
find "$VC2_ROOT/vc2_dpo/raw" -type f -name '*.mp4' | wc -l
find "$VC2_ROOT" -path '*vbench_standard_named*' -type f -name '*.mp4' | wc -l
```

fullmask：

```bash
find "$FM_ROOT/diffueraser_base/vbench_standard_named" -type f -name '*.mp4' | wc -l
find "$FM_ROOT/diffueraser_dpo/vbench_standard_named" -type f -name '*.mp4' | wc -l
```

#### 19.7.4 日志层

真实错误关键词：

```text
Traceback
RuntimeError
CUDA out of memory
out of memory
No such file
ModuleNotFoundError
FileNotFoundError
Killed
failed
TypeError
ValueError
```

应过滤掉的非致命提示：

```text
FutureWarning
UserWarning
warning
savefps type already patched
video writer already patched
multi-gpu env ranks already patched
You are using a model of type clip_text_model
not supported for all configurations
will be ignored
deprecated
```

`You are using a model of type clip_text_model...` 是 transformers 兼容提示。它包含 `errors` 字样时容易被 naive grep 误报，但不是失败。

### 19.8 已经踩过的坑与解决方案

#### 19.8.1 PAI git pull 不稳定

症状：

```text
RPC failed; curl 52 Empty reply from server
GnuTLS recv error (-110)
Failed to connect to github.com port 443
fatal: expected acknowledgments
```

解决：

```bash
git config --global http.version HTTP/1.1
git config --global http.postBuffer 524288000
for i in 1 2 3 4 5 6 7 8 9 10; do
  git pull --ff-only origin main && break
  sleep 10
done
```

如果仍失败：

```text
在 H20 拉最新，然后 PAI 从 H20 rsync 小脚本。
```

#### 19.8.2 rsync / SSH reset

症状：

```text
kex_exchange_identification: read: Connection reset by peer
ssh_dispatch_run_fatal: Broken pipe
rsync error code 255
```

解决：

```bash
rsync -ahP --partial --append-verify \
  -e "ssh -p 22 -o ServerAliveInterval=30 -o ServerAliveCountMax=120 -o ConnectTimeout=30" \
  SRC DST
```

大文件中断后可重试，`--append-verify` 会续传并校验。

#### 19.8.3 tee 日志目录不存在

症状：

```text
tee: ... No such file or directory
```

原因：

```text
tee 在 wrapper mkdir 前先打开日志文件。
```

解决：

```bash
OUT_ROOT=...
mkdir -p "$OUT_ROOT"
bash wrapper.sh 2>&1 | tee "$OUT_ROOT/log.txt"
```

#### 19.8.4 AUDIT_LOG unbound variable

症状：

```text
bash: line 137: AUDIT_LOG: unbound variable
```

原因：

```text
heredoc 内部 set -u，但 AUDIT_LOG 没 export，或粘贴时 heredoc 末尾被污染。
```

解决：

```bash
AUDIT_LOG=/tmp/pai_two_eval_audit_$(date +%Y%m%d_%H%M%S).log
export AUDIT_LOG
```

并保证：

```text
BASH
```

单独成行。

该错误只影响审查脚本最后 echo，不影响正在跑的评测。

#### 19.8.5 fullmask PyAV codec None

症状：

```text
TypeError: expected bytes, NoneType found
```

位置：

```text
tools/generate_diffueraser_fullmask_vbench.py
imageio.mimsave
```

解决：

```text
commit be96042
explicit codec="libx264"
fallback cv2 mp4v writer
```

#### 19.8.6 VideoDPO PyAV / torchvision writer

症状：

```text
AttributeError: 'str' object has no attribute 'numerator'
TypeError: an integer is required
```

解决：

```text
savefps type int
imageio/cv2 fallback writer
```

#### 19.8.7 prompt limit 与 full VBench 混淆

错误用法：

```text
PROMPT_LIMIT=2 with RUN_VBENCH=1
PROMPT_RANDOM_LIMIT=30 with RUN_VBENCH=1
```

正确：

```text
smoke:
  RUN_VBENCH=0
  PROMPT_LIMIT=2 or PROMPT_RANDOM_LIMIT=30
  SAMPLES_PER_PROMPT=1

full quantitative:
  RUN_VBENCH=1
  PROMPT_LIMIT=0
  no PROMPT_RANDOM_LIMIT
  SAMPLES_PER_PROMPT=5
  full vbench_standard_prompts.txt
```

#### 19.8.8 fullmask baseline/DPO 不能加 ProPainter prior

风险：

```text
把 fullmask-DiffuEraser 误写成 ProPainter prior + DiffuEraser。
```

正确实验定义：

```text
masked images: blank black images
masks: full masks
prompt: VBench prompt
model: DiffuEraser architecture
baseline weights: original converted DiffuEraser
DPO weights: fullmask-DPO last_weights
```

#### 19.8.9 首帧条件混淆

风险：

```text
把 VC2/DiffuEraser 评测写成需要首帧。
```

当前实际：

```text
ordinary VideoDPO VC2: text-to-video style prompt generation through VideoDPO inference.
fullmask DiffuEraser: mask + masked images + prompt; no real first-frame condition.
```

DiffuEraser paper 图 2 中输入是 masks 和 masked images，不是首帧。

#### 19.8.10 official VideoDPO DPO diagnostics 不等于 fullmask diagnostics

official VideoDPO DPO diagnostics 来自 H20：

```text
/home/nvme01/H20_Video_inpainting_DPO/logs/videodpo_vc2_dpo/...
```

fullmask DPO diagnostics 来自 PAI training run：

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260516_024051_lingbot-world-model-fullmask-videodpo-epoch5-gpu4-7-20260516_104046
```

两者不能混用。official VideoDPO diagnostics 用于验证复现训练行为；fullmask diagnostics 用于验证本项目 bridge training 行为。

### 19.9 当前最终应收集的结果清单

#### 19.9.1 ordinary VideoDPO full quantitative

等待终端 1 完成后，应收集：

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_full/pai_vc2_vs_dpo_full_20260517_034228/full_vbench.log
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_full/pai_vc2_vs_dpo_full_20260517_034228/vc2_base/vbench_eval/summary.json
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_full/pai_vc2_vs_dpo_full_20260517_034228/vc2_base/vbench_eval/summary.csv
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_full/pai_vc2_vs_dpo_full_20260517_034228/vc2_dpo/vbench_eval/summary.json
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_full/pai_vc2_vs_dpo_full_20260517_034228/vc2_dpo/vbench_eval/summary.csv
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_full/pai_vc2_vs_dpo_full_20260517_034228/vbench_paper_table.md
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_full/pai_vc2_vs_dpo_full_20260517_034228/vbench_paper_table.csv
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_full/pai_vc2_vs_dpo_full_20260517_034228/vbench_paper_table.tex
```

#### 19.9.2 fullmask DiffuEraser full quantitative

等待终端 2 完成后，应收集：

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_035156/diffueraser_base.log
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_035156/diffueraser_dpo.log
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_035156/diffueraser_base/vbench_eval/summary.json
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_035156/diffueraser_base/vbench_eval/summary.csv
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_035156/diffueraser_dpo/vbench_eval/summary.json
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_035156/diffueraser_dpo/vbench_eval/summary.csv
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_035156/vbench_paper_table.md
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_035156/vbench_paper_table.csv
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_035156/vbench_paper_table.tex
```

#### 19.9.3 定性结果

如果还未运行，应后续单独启动 30 prompt qualitative smoke。不要从 full quantitative 中手工挑视频冒充定性结果，最好使用固定 seed prompt subset，并输出 side-by-side folder。

ordinary qualitative expected：

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_qual_smoke/<run>/qual_side_by_side/*.mp4
```

fullmask qualitative expected：

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/fullmask_qual_smoke/<run>/qual_side_by_side/*.mp4
```

### 19.10 下一步工程建议

1. fullmask 多 GPU sharding：

```text
当前 fullmask full VBench 很慢，因为 baseline/DPO 各单卡。
建议扩展 generate_diffueraser_fullmask_vbench.py:
  --rank
  --world_size
  --device_index optional
再扩展 wrapper 启动多 rank。
```

2. 将 audit 脚本固化为 repo tool：

```text
tools/pai_vbench_runtime_audit.sh
```

避免每次 heredoc 粘贴污染，尤其避免 `AUDIT_LOG unbound variable`。

3. 将 fullmask base/DPO paired runner 固化：

```text
DPO_finetune/scripts/sc_diffueraser_fullmask_vbench_pair.sbatch
```

当前是用户手动在终端 2 启动两条 background 命令，容易漏掉 DPO 或只跑 baseline。

4. 增加 output manifest：

```text
每个 full eval 结束后自动写 manifest.json:
  git_commit
  start_time
  end_time
  prompt_file sha256
  checkpoint paths
  env paths
  mp4 counts
  summary paths
```

5. HPS / PickScore：

```text
如果要和 VideoDPO.pdf 表完全一致，需要独立评估 HPS(V) 与 PickScore。
当前 table 中这两列保留 "-"，这是诚实输出。
```

6. process title：

```text
fullmask 已经显示 lingbot-world-model。
ordinary VideoDPO external inference 仍可能显示 /mnt/nas/hj/conda_envs/videodpo/bin/python。
如果必须强制 nvidia-smi 也显示 lingbot-world-model，需要确保 external VideoDPO inference 进程启动时也 import sitecustomize/setproctitle 并覆盖 argv/title。
```

## 20. 2026-05-17 续写：完整交接版记录、代码构思、整体框架和高风险易错点

本节是在前面第 18、19 节基础上继续追加，不改动原有内容。目标是把本轮到目前为止所有关键上下文写成“下一次打开就能继续干”的交接材料，尤其包括：

```text
1. PAI / H20 / Git / env / checkpoint 的当前事实。
2. ordinary VideoDPO VC2 base vs DPO 的定量与定性评测框架。
3. fullmask DiffuEraser baseline vs fullmask DiffuEraser DPO 的定量与定性评测框架。
4. 已经写入 repo 的代码模块、它们的设计目的、输入输出约定。
5. 目前正在 PAI 上跑的两个终端的状态。
6. 容易误判、容易复制错、容易把 full run 当 smoke 的坑。
```

### 20.1 当前机器和仓库事实

本地 `/home/hj/Video_inpainting_DPO` 是开发侧仓库，当前本地可见的最新提交序列是：

```text
be96042 Fix fullmask VBench video writer
9160ab4 Add prompt sampling helper
65eefe0 Add VBench named side-by-side helper
ea946f2 Add multi GPU qualitative VBench smoke
f5c56db Add paper-style VBench table export
904eda7 Patch VideoDPO writer for PyAV compatibility
589d1f8 Patch VideoDPO savefps for VBench smoke
cdbc7bc Use explicit Python for VideoDPO VBench
```

PAI 侧实际执行仓库是：

```text
/mnt/nas/hj/H20_Video_inpainting_DPO
```

PAI 最近审查输出中显示：

```text
hostname: dsw-753014-dc85766cb-4v2jj
audit time: Sun May 17 11:24:06 AM CST 2026
git commit: be96042
```

本地当前环境无法直接访问 `/mnt/nas/hj/H20_Video_inpainting_DPO`，因此本节 PAI runtime 事实来自用户粘贴的审查日志，而不是本地实时重新扫描。

### 20.2 不要动的本地脏状态

本地仓库已有一些与本轮任务无关的删除或未跟踪文件。这些不要回滚、不要自动修复，除非用户明确要求：

```text
D PRD/First_Finetuning_Summary.md
D PRD/PPT.pptx
D PRD/stage2_motion_module_init.md
D PRD/update_ppt_tables.py
D PRD/validation_optimization.md
?? videodpo-vc2-dpo-diag-log/
```

本轮只持续追加这个 runbook：

```text
PRD/PAI_H20_VideoDPO_Migration_Runbook_20260515.md
```

### 20.3 资产和环境总表

PAI VideoDPO env：

```text
/mnt/nas/hj/conda_envs/videodpo
```

这个 env 是从 H20 打包、rsync 到 PAI 后解压得到的：

```text
H20 pack:
/home/nvme01/H20_Video_inpainting_DPO/data/hf_upload/videodpo_env_for_pai.tar.gz

PAI pack:
/mnt/nas/hj/env_packs/videodpo_env_for_pai.tar.gz

PAI unpack target:
/mnt/nas/hj/conda_envs/videodpo
```

校验曾经通过：

```text
sha256sum -c videodpo_env_for_pai.tar.gz.sha256
videodpo_env_for_pai.tar.gz: OK
```

VideoDPO env 中关键依赖已确认：

```text
torch OK
pytorch_lightning OK
omegaconf OK
einops OK
decord OK
av OK
imageio OK
cv2 OK
transformers OK
open_clip OK
kornia OK
fairscale OK
peft OK
setproctitle installed later
```

PAI DiffuEraser env：

```text
/mnt/nas/hj/conda_envs/diffueraser
```

ordinary VideoDPO / VC2 checkpoints：

```text
VC2 base:
/mnt/nas/hj/H20_Video_inpainting_DPO/external/VideoDPO/checkpoints/vc2/model.ckpt

VC2 ref model:
/mnt/nas/hj/H20_Video_inpainting_DPO/external/VideoDPO/checkpoints/vc2/ref_model.ckpt

VC2 DPO checkpoint:
/mnt/nas/hj/h20_vbench_assets/checkpoints/vc2_dpo_last.ckpt
```

fullmask DiffuEraser checkpoints：

```text
fullmask baseline:
/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000

fullmask DPO trained output:
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260516_024051_lingbot-world-model-fullmask-videodpo-epoch5-gpu4-7-20260516_104046/last_weights
```

fullmask DPO 的 `last_weights` 下应包含：

```text
unet_main/config.json
unet_main/diffusion_pytorch_model.safetensors
brushnet/config.json
brushnet/diffusion_pytorch_model.safetensors
```

### 20.4 baseline 到底是谁

ordinary VideoDPO 复现线：

```text
baseline = official VideoDPO VC2 base model.ckpt
DPO      = H20 训练得到的 vc2_dpo_last.ckpt
```

fullmask DiffuEraser 线：

```text
baseline = original DiffuEraser converted_weights_step48000
DPO      = fullmask DiffuEraser DPO stage1 last_weights
```

fullmask 这条线不是 original VideoDPO baseline，也不是 ProPainter + VideoDPO baseline。它的输入设计是 full mask / blank masked image / prompt condition，目的是验证“把 VideoDPO 偏好学习迁移到 DiffuEraser fullmask 视频生成/补全设定”。

### 20.5 最终必须交付的四类东西

用户明确要求最终要有四个结果：

```text
1. ordinary VideoDPO VC2 base vs VC2 DPO 的定性二合一视频文件夹。
2. fullmask DiffuEraser baseline vs fullmask DiffuEraser DPO 的定性二合一视频文件夹。
3. ordinary VideoDPO VC2 base vs VC2 DPO 的 VBench paper-style 表格。
4. fullmask DiffuEraser baseline vs fullmask DiffuEraser DPO 的 VBench paper-style 表格。
```

定量 VBench 必须使用完整 prompt txt：

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/external/VideoDPO/prompts/vbench_standard_prompts.txt
```

不要把 `PROMPT_RANDOM_LIMIT=30` 的结果当作定量结果。随机 30 prompt 只用于定性 smoke / qualitative side-by-side 快速看效果。

### 20.6 ordinary VideoDPO 评测整体框架

ordinary VideoDPO 使用主 wrapper：

```text
DPO_finetune/scripts/sc_videodpo_vc2_vbench.sbatch
```

核心输入：

```text
CONDA_ENV=/mnt/nas/hj/conda_envs/videodpo
CKPT_SPECS=vc2_base:<base_ckpt>,vc2_dpo:<dpo_ckpt>
OUT_ROOT=<run_output_root>
SAMPLES_PER_PROMPT=5
RUN_INFERENCE=1
RUN_VBENCH=1
MAKE_PAPER_TABLE=1
NUM_INFER_GPUS=4
CUDA_VISIBLE_DEVICES=0,1,2,3
```

执行逻辑：

```text
1. wrapper 使用 CONDA_ENV/bin/python，不依赖当前 shell active env。
2. 进入 external/VideoDPO，调用 scripts/inference.py。
3. 对每个 label:
   - vc2_base
   - vc2_dpo
4. 对每个 sample:
   - sample_0 seed=20230211
   - sample_1 seed=20230212
   - sample_2 seed=20230213
   - sample_3 seed=20230214
   - sample_4 seed=20230215
5. 每个 sample 使用完整 946 prompts。
6. NUM_INFER_GPUS=4 时，每轮 sample 会起 4 个 rank 进程。
7. raw 输出是数字文件名：
   raw/sample_<k>/0000.mp4 ...
8. prepare-vbench 将 raw 数字文件转换为 VBench 标准命名：
   vbench_standard_named/<prompt>-<sample>.mp4
9. RUN_VBENCH=1 后调用 VBench evaluate。
10. MAKE_PAPER_TABLE=1 后生成 paper-style 表。
```

ordinary VideoDPO 预期定量输出：

```text
<VC2_ROOT>/vc2_base/vbench_eval/summary.json
<VC2_ROOT>/vc2_base/vbench_eval/summary.csv
<VC2_ROOT>/vc2_dpo/vbench_eval/summary.json
<VC2_ROOT>/vc2_dpo/vbench_eval/summary.csv
<VC2_ROOT>/vbench_paper_table.md
<VC2_ROOT>/vbench_paper_table.csv
<VC2_ROOT>/vbench_paper_table.tex
```

ordinary VideoDPO 预期定性输出：

```text
<QUAL_ROOT>/qual_side_by_side/*.mp4
```

side-by-side 工具：

```text
tools/videodpo_make_side_by_side.py
```

### 20.7 fullmask DiffuEraser 评测整体框架

fullmask DiffuEraser 使用 wrapper：

```text
DPO_finetune/scripts/h20_diffueraser_fullmask_vbench.sh
```

核心生成脚本：

```text
tools/generate_diffueraser_fullmask_vbench.py
```

fullmask 评测被拆成两个 background job：

```text
job A:
  CUDA_VISIBLE_DEVICES=4
  WEIGHTS_PATH=/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000
  OUT_ROOT=<FM_ROOT>/diffueraser_base

job B:
  CUDA_VISIBLE_DEVICES=5
  WEIGHTS_PATH=/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/.../last_weights
  OUT_ROOT=<FM_ROOT>/diffueraser_dpo
```

注意：用户希望“终端 2 使用 4-7 号 GPU”，但当前代码实现不是 4 GPU sharding，而是 baseline 一个进程占 GPU4，DPO 一个进程占 GPU5。GPU6/7 空闲在当前实现下是正常的，不代表没跑；只是没有做到 fullmask 多卡 prompt sharding。

fullmask 生成逻辑：

```text
1. 读取完整 vbench_standard_prompts.txt。
2. 对每个 sample seed 生成一遍完整 946 prompts。
3. 每个模型总生成 946 * 5 = 4730 个 mp4。
4. 直接输出为 VBench 标准命名：
   vbench_standard_named/<prompt>-<sample>.mp4
5. 生成完成后 RUN_VBENCH=1 进入 VBench evaluate。
6. 两边 summary 生成后，再用 tools/vbench_make_paper_table.py 生成统一表。
```

fullmask 预期定量输出：

```text
<FM_ROOT>/diffueraser_base/vbench_eval/summary.json
<FM_ROOT>/diffueraser_base/vbench_eval/summary.csv
<FM_ROOT>/diffueraser_dpo/vbench_eval/summary.json
<FM_ROOT>/diffueraser_dpo/vbench_eval/summary.csv
<FM_ROOT>/vbench_paper_table.md
<FM_ROOT>/vbench_paper_table.csv
<FM_ROOT>/vbench_paper_table.tex
```

fullmask 预期定性输出：

```text
<FULLMASK_QUAL_ROOT>/qual_side_by_side/*.mp4
```

side-by-side 工具：

```text
tools/vbench_named_side_by_side.py
```

### 20.8 当前正在跑的 PAI 两个终端状态

以下来自用户在 2026-05-17 11:24 CST 粘贴的审查输出。

#### 20.8.1 终端 1：ordinary VideoDPO VC2 base vs VC2 DPO full VBench

run root：

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_full/pai_vc2_vs_dpo_full_20260517_034228
```

GPU：

```text
CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_INFER_GPUS=4
```

进程：

```text
parent:
bash DPO_finetune/scripts/sc_videodpo_vc2_vbench.sbatch

children:
/mnt/nas/hj/conda_envs/videodpo/bin/python scripts/inference.py ...
```

当时状态：

```text
current label: vc2_base
current sample: sample_4
current seed: 20230215
vc2_base raw mp4: 4176 / 4730
vc2_dpo raw mp4: 0 / 4730
vbench named mp4: 0 in audit summary, but main log showed prepare-vbench completed for sample_0..3
```

已经完成的 prepare-vbench：

```text
sample_0 written=946 missing=0
sample_1 written=946 missing=0
sample_2 written=946 missing=0
sample_3 written=946 missing=0
```

判断：

```text
普通 VideoDPO full VBench 正常跑着。
还没有开始 vc2_dpo，是因为 vc2_base 五个 samples 还没完全生成完。
没有可疑错误。
```

#### 20.8.2 终端 2：fullmask DiffuEraser baseline vs DPO full VBench

run root：

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_035156
```

进程 A：

```text
CUDA_VISIBLE_DEVICES=4
OUT_ROOT=<FM_ROOT>/diffueraser_base
WEIGHTS_PATH=/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000
```

进程 B：

```text
CUDA_VISIBLE_DEVICES=5
OUT_ROOT=<FM_ROOT>/diffueraser_dpo
WEIGHTS_PATH=/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260516_024051_lingbot-world-model-fullmask-videodpo-epoch5-gpu4-7-20260516_104046/last_weights
```

当时状态：

```text
diffueraser_base mp4: 2512 / 4730
diffueraser_dpo mp4: 2513 / 4730
current sample: sample_2
current prompt position: around 600 / 946
```

判断：

```text
fullmask DiffuEraser full VBench 正常跑着。
GPU4/GPU5 忙，GPU6/GPU7 空闲是当前 wrapper 设计导致，不是错误。
没有可疑错误。
```

### 20.9 代码构思和文件责任

#### 20.9.1 sc_videodpo_vc2_vbench.sbatch

文件：

```text
DPO_finetune/scripts/sc_videodpo_vc2_vbench.sbatch
```

设计目的：

```text
把 official VideoDPO VC2 checkpoint 和 DPO checkpoint 放到同一个 VBench 复现框架里跑。
既支持 full quantitative，也支持 random 30 qualitative smoke。
避免依赖 slurm，PAI 上可以 bash 直接运行。
```

关键设计点：

```text
1. CONDA_ENV 显式指定 Python。
2. CKPT_SPECS 支持多个 label:path，用逗号分隔。
3. NUM_INFER_GPUS 支持多 rank inference。
4. PROMPT_LIMIT 用于小规模 smoke。
5. PROMPT_RANDOM_LIMIT 用于固定 seed 随机抽 prompt。
6. MAKE_QUAL_COMPARE 调用 side-by-side 工具。
7. MAKE_PAPER_TABLE 调用 VBench paper table 工具。
8. runtime patch external VideoDPO inference.py，让每个 rank 只处理自己 shard 的 prompts。
9. runtime patch savefps / video writer，绕开 PyAV 和 fps 类型问题。
```

#### 20.9.2 vbench_make_paper_table.py

文件：

```text
tools/vbench_make_paper_table.py
```

设计目的：

```text
读取多个 VBench summary.json / summary.csv，
整理成接近 VideoDPO.pdf 中表格布局的 Markdown / CSV / LaTeX。
```

重要约定：

```text
VBench 可以算 quality / semantic / subject / background / motion / aesthetics 等维度。
HPS / PickScore 当前没有被 VBench evaluate 自动产生。
所以 paper table 中 HPS / PickScore 应显示 "-"，不能伪造数值。
```

#### 20.9.3 videodpo_make_side_by_side.py

文件：

```text
tools/videodpo_make_side_by_side.py
```

设计目的：

```text
用于 ordinary VideoDPO raw 输出。
把 vc2_base/raw/sample_*/0001.mp4 这类数字文件，
和 vc2_dpo/raw/sample_*/0001.mp4 配对，
生成横向二合一视频，并在视频上标注 label。
```

适合场景：

```text
ordinary VideoDPO qualitative random30 smoke
```

#### 20.9.4 vbench_named_side_by_side.py

文件：

```text
tools/vbench_named_side_by_side.py
```

设计目的：

```text
用于已经是 VBench 标准命名的 mp4。
例如：
  <prompt>-0.mp4
  <prompt>-1.mp4
它可以将两个 named folders 里的同名视频配对生成 side-by-side。
```

适合场景：

```text
fullmask DiffuEraser qualitative
ordinary VideoDPO prepare-vbench 之后的 named qualitative
```

#### 20.9.5 sample_prompts.py

文件：

```text
tools/sample_prompts.py
```

设计目的：

```text
从完整 VBench prompt txt 中用固定 seed 抽 30 条 prompt。
只用于 qualitative smoke。
```

不能用于：

```text
正式 VBench 定量表。
```

#### 20.9.6 generate_diffueraser_fullmask_vbench.py

文件：

```text
tools/generate_diffueraser_fullmask_vbench.py
```

设计目的：

```text
用 DiffuEraser pipeline 按 VBench prompt 生成 fullmask 视频，
并直接保存为 VBench 标准命名。
```

关键修复：

```text
旧版 imageio / PyAV 在 PAI 上保存 mp4 会报：
TypeError: expected bytes, NoneType found

be96042 后保存逻辑改成：
1. 先 imageio.mimsave(path, arrays, fps=fps, codec="libx264", macro_block_size=1)
2. 如果失败，再 fallback 到 cv2.VideoWriter + mp4v
```

这个修复已经通过后续 fullmask 长跑验证：新的 run 没有再出现该 TypeError。

### 20.10 最容易出错的点

#### 20.10.1 git pull 失败不等于代码错

PAI 上多次出现：

```text
error: RPC failed; curl 52 Empty reply from server
fatal: expected 'acknowledgments'
Failed to connect to github.com port 443
GnuTLS recv error (-110)
```

判断：

```text
这是 GitHub 网络 / TLS / 连接稳定性问题，不是 repo 内容问题。
之前同一机器能 pull，后来不能 pull，说明网络链路不稳定。
```

常用缓解：

```text
git config --global http.version HTTP/1.1
git config --global http.postBuffer 524288000
for i in 1 2 3 4 5 6 7 8 9 10; do git pull --ff-only origin main && break; sleep 10; done
```

#### 20.10.2 heredoc 粘贴污染

用户多次在终端中看到命令互相覆盖、字符粘连，例如：

```text
BASH "Audit log: $AUDIT_LOG"
AUDIT_LOG: unbound variable
```

判断：

```text
这通常是复制 heredoc 时末尾/下一行混进 shell 的问题。
只要核心审查主体输出完整，最后这行 unbound variable 不代表评测失败。
```

建议：

```text
后续把 audit 写成 tools/pai_vbench_runtime_audit.sh，
不要继续手粘 100+ 行 heredoc。
```

#### 20.10.3 tee 之前必须 mkdir

之前 qualitative smoke 有：

```text
tee: .../xxx.smoke.log: No such file or directory
```

原因：

```text
OUT_ROOT 的父目录未提前创建。
```

修复方式：

```text
mkdir -p "$OUT_ROOT"
```

或者至少：

```text
mkdir -p "$(dirname "$OUT_ROOT")"
```

#### 20.10.4 full VBench 和 random30 qualitative 不要混

full quantitative：

```text
PROMPT_LIMIT=0
PROMPT_RANDOM_LIMIT unset
SAMPLES_PER_PROMPT=5
RUN_VBENCH=1
完整 vbench_standard_prompts.txt
```

qualitative smoke：

```text
PROMPT_RANDOM_LIMIT=30
PROMPT_RANDOM_SEED=20260517
SAMPLES_PER_PROMPT=1
RUN_VBENCH=0
MAKE_QUAL_COMPARE=1
```

不要用 random30 结果做论文表格。

#### 20.10.5 GPU 编号语义

ordinary VideoDPO：

```text
CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_INFER_GPUS=4
```

进程内部看到的是：

```text
cuda:0, cuda:1, cuda:2, cuda:3
```

它们对应物理 GPU：

```text
0,1,2,3
```

fullmask DiffuEraser：

```text
baseline CUDA_VISIBLE_DEVICES=4
DPO      CUDA_VISIBLE_DEVICES=5
```

进程内部都可能只看见 `cuda:0`，但 nvidia-smi 物理卡显示是 GPU4 / GPU5。

#### 20.10.6 fullmask 没用 GPU6/7 不是当前错误

用户希望终端 2 用 4-7。当前实现只启动两个模型进程：

```text
baseline -> GPU4
DPO      -> GPU5
```

所以 GPU6/7 空闲。要真正用 4-7，需要新增 fullmask prompt sharding：

```text
generate_diffueraser_fullmask_vbench.py:
  --rank
  --world_size

h20_diffueraser_fullmask_vbench.sh or new pair wrapper:
  baseline ranks on GPU4,6
  dpo ranks on GPU5,7
```

当前先不改动正在跑的 job。

#### 20.10.7 process name 只是一层安全标识

用户要求进程名保持：

```text
lingbot-world-model
```

用途：

```text
区分自己任务和别人任务。
清理 GPU 残留时只能动明确带这个标识且 env/path/OUT_ROOT 都属于本项目的进程。
```

注意：

```text
ordinary VideoDPO external inference 在 nvidia-smi 中可能仍显示 /mnt/nas/hj/conda_envs/videodpo/bin/python。
这不一定是错，因为 external process title 未必被 setproctitle 覆盖。
要判断归属必须看 /proc/<pid>/environ 中：
  WORLDMODELPHY_PROCESS_NAME
  PROCESS_TITLE
  OUT_ROOT
  CKPT_SPECS
  PROJECT_ROOT
```

#### 20.10.8 不要误杀别人的任务

只有同时满足以下条件才可以考虑清理：

```text
1. ps / environ 里能看到 WORLDMODELPHY_PROCESS_NAME=lingbot-world-model。
2. cwd 在 /mnt/nas/hj/H20_Video_inpainting_DPO 或 external/VideoDPO。
3. OUT_ROOT 属于本次 logs/vbench_* run。
4. CKPT_SPECS 或 WEIGHTS_PATH 是本项目路径。
5. 用户明确说可以清理残留。
```

否则不要 kill。

之前 `kill -TERM "-$PGID"` 因粘贴污染或 PGID 解析问题出现过：

```text
arguments must be process or job IDs
```

后续清理最好先打印：

```text
PID=<pid>
PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
echo "PID=$PID PGID=$PGID"
ps -o pid,ppid,pgid,args -g "$PGID"
```

确认无误再 kill。

#### 20.10.9 警告不一定是错误

审查日志中应忽略这些已知 warning：

```text
FutureWarning
UserWarning
savefps type already patched
video writer already patched
multi-gpu env ranks already patched
You are using a model of type clip_text_model
not supported for all configurations
will be ignored
deprecated
```

真正要关注：

```text
Traceback
RuntimeError
CUDA out of memory
No such file or directory
ModuleNotFoundError
TypeError: expected bytes, NoneType found
missing > 0
rsync error
```

#### 20.10.10 VBench 表格不要伪造 HPS / PickScore

VideoDPO.pdf 论文表可能有 HPS / PickScore 一类列。当前本项目自动 VBench evaluate 不产生这些指标，因此：

```text
HPS(V) = "-"
PickScore = "-"
```

这是正确的诚实占位。要填真实值，需要另开 evaluator。

### 20.11 审查状态如何读

ordinary VideoDPO 现在看到：

```text
vc2_base raw mp4 > 0
vc2_dpo raw mp4 = 0
```

这不表示 DPO 没跑起来。因为 wrapper 顺序是先跑完整 `vc2_base` 的 5 samples，再跑 `vc2_dpo`。

当 ordinary VideoDPO 完整生成结束后，应该看到：

```text
vc2_base raw mp4 = 4730
vc2_dpo raw mp4 = 4730
vc2_base prepare-vbench written=946 missing=0 for sample_0..4
vc2_dpo prepare-vbench written=946 missing=0 for sample_0..4
```

fullmask 现在看到：

```text
diffueraser_base mp4 ~= diffueraser_dpo mp4
```

两边数量差 1 或很小通常是正常的，因为两个独立进程速度略有不同。

fullmask 完整生成结束后，应该看到：

```text
diffueraser_base mp4 = 4730
diffueraser_dpo mp4 = 4730
```

然后才进入 VBench evaluate 和 table 生成。

### 20.12 当前两个 full quantitative run 的预期完成路径

ordinary VideoDPO：

```text
VC2_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_full/pai_vc2_vs_dpo_full_20260517_034228

expected final:
$VC2_ROOT/vbench_paper_table.md
$VC2_ROOT/vbench_paper_table.csv
$VC2_ROOT/vbench_paper_table.tex
```

fullmask DiffuEraser：

```text
FM_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_035156

expected final:
$FM_ROOT/vbench_paper_table.md
$FM_ROOT/vbench_paper_table.csv
$FM_ROOT/vbench_paper_table.tex
```

如果 table 没自动出现，但 summary 已经出现，可以手动生成：

```bash
cd /mnt/nas/hj/H20_Video_inpainting_DPO

python tools/vbench_make_paper_table.py \
  --input vc2_base:$VC2_ROOT/vc2_base/vbench_eval/summary.json \
  --input vc2_dpo:$VC2_ROOT/vc2_dpo/vbench_eval/summary.json \
  --out-prefix $VC2_ROOT/vbench_paper_table
```

fullmask：

```bash
cd /mnt/nas/hj/H20_Video_inpainting_DPO

python tools/vbench_make_paper_table.py \
  --input diffueraser_base:$FM_ROOT/diffueraser_base/vbench_eval/summary.json \
  --input diffueraser_dpo:$FM_ROOT/diffueraser_dpo/vbench_eval/summary.json \
  --out-prefix $FM_ROOT/vbench_paper_table
```

### 20.13 定性 qualitative 的推荐后续

定性只跑 30 个固定随机 VBench prompts。不要用完整 full quantitative 的全部视频来生成全部 side-by-side，否则文件太多、时间太长。

ordinary VideoDPO qualitative 设计：

```text
PROMPT_RANDOM_LIMIT=30
PROMPT_RANDOM_SEED=20260517
SAMPLES_PER_PROMPT=1
RUN_INFERENCE=1
RUN_VBENCH=0
MAKE_QUAL_COMPARE=1
CUDA_VISIBLE_DEVICES 可用空闲卡
```

fullmask qualitative 设计：

```text
使用 tools/sample_prompts.py 抽 30 prompt。
分别生成 diffueraser_base 和 diffueraser_dpo。
用 tools/vbench_named_side_by_side.py 配对。
```

输出目标：

```text
ordinary:
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_qual_smoke/<run>/qual_side_by_side

fullmask:
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/fullmask_qual_smoke/<run>/qual_side_by_side
```

### 20.14 如果下次继续接手，先做什么

先不要启动新 job。先跑审查，确认两个 full quantitative 是否已完成：

```bash
cd /mnt/nas/hj/H20_Video_inpainting_DPO || exit 1

VC2_ROOT=$(ls -td logs/vbench_full/pai_vc2_vs_dpo_full_* 2>/dev/null | head -1)
FM_ROOT=$(ls -td logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_* 2>/dev/null | head -1)

echo "VC2_ROOT=$VC2_ROOT"
echo "FM_ROOT=$FM_ROOT"

nvidia-smi

find "$VC2_ROOT" -type f -name '*.mp4' | wc -l
find "$FM_ROOT/diffueraser_base/vbench_standard_named" -type f -name '*.mp4' | wc -l
find "$FM_ROOT/diffueraser_dpo/vbench_standard_named" -type f -name '*.mp4' | wc -l

find "$VC2_ROOT" -type f \( -name 'summary.json' -o -name 'summary.csv' -o -name 'vbench_paper_table.md' \) -print
find "$FM_ROOT" -type f \( -name 'summary.json' -o -name 'summary.csv' -o -name 'vbench_paper_table.md' \) -print
```

如果 GPU 已空且 summary/table 都出现：

```text
1. 打开 vbench_paper_table.md 看 ordinary 表。
2. 打开 fullmask vbench_paper_table.md 看 fullmask 表。
3. 再跑 qualitative random30，生成两个 side-by-side folders。
4. 把最终四个路径写回 PRD。
```

如果 GPU 还在跑且 mp4 数量持续增长：

```text
不要干预，继续等。
```

如果 GPU 在跑但 mp4 数量长时间不增长：

```text
1. tail 对应 log。
2. 查 Traceback / RuntimeError / CUDA OOM。
3. 确认不是正在 VBench evaluate 阶段，因为 evaluate 阶段不会继续增加 mp4。
```

### 20.15 对当前进度的结论

截至 2026-05-17 11:24 CST 的审查：

```text
ordinary VideoDPO full VBench:
  正常。
  GPU0-3 满载。
  正在跑 vc2_base sample_4。
  还没开始 vc2_dpo。
  没有可疑错误。

fullmask DiffuEraser full VBench:
  正常。
  GPU4/GPU5 满载。
  baseline 和 DPO 都在 sample_2，进度大约 2500+/4730。
  GPU6/GPU7 空闲是当前未做 sharding 的设计限制。
  没有可疑错误。
```

最重要的后续判断点：

```text
1. ordinary 等 vc2_base 4730 完成后，会自动开始 vc2_dpo。
2. fullmask 等 base/dpo 都 4730 后，会进入 VBench evaluate。
3. 最终不要只看 mp4，要看 summary.json + vbench_paper_table.md。
4. 定量用 full txt，定性用 random30。
```

## 21. 2026-05-18 续写：两个 PAI 终端停止后的失败审查与修复方向

本节记录 2026-05-18 12:12 CST 在 PAI 上对两个长跑终端的审查结果。结论是：两个终端都已经停止，GPU 全空，但不是正常完成；两个 run 都在 VBench evaluation 阶段失败或被中断，最终 paper-style VBench 表格没有产出。

### 21.1 审查上下文

审查命令输出：

```text
hostname: dsw-753014-dc85766cb-4v2jj
time: Mon May 18 12:12:45 PM CST 2026
git commit: be96042
audit log: /tmp/pai_two_terminal_done_check_20260518_121245.log
```

latest roots：

```text
VC2_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_full/pai_vc2_vs_dpo_full_20260517_034228
FM_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_035156
```

GPU 状态：

```text
GPU0-7 memory all 0 MiB
No running processes found
```

运行进程状态：

```text
No lingbot-world-model / sc_videodpo_vc2_vbench / h20_diffueraser_fullmask_vbench /
scripts/inference.py / generate_diffueraser_fullmask_vbench / evaluate.py processes.
```

判断：

```text
两个终端都已经停止。
但停止不等于正常完成，因为后续输出和日志显示失败。
```

### 21.2 ordinary VideoDPO VC2 run 当前状态

审查输出：

```text
vc2_base raw mp4   = 4730 / 4730
vc2_dpo raw mp4    = 0 / 4730
vc2_base named mp4 = 0 / 4730
vc2_dpo named mp4  = 0 / 4730
```

关键失败日志：

```text
ERROR 618: jwt:expired.

subprocess.CalledProcessError:
Command '['wget',
'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth',
'-P', '/root/.cache/vbench/pyiqa_model']'
returned non-zero exit status 8.
```

原因判断：

```text
ordinary VideoDPO wrapper 的执行顺序是顺序式：
1. 先生成 vc2_base 的所有 samples。
2. prepare-vbench。
3. 对 vc2_base 跑 VBench eval。
4. 再进入 vc2_dpo 生成和 eval。

当前 vc2_base 生成已经完成 4730/4730。
但 vc2_base 的 VBench eval 下载 MUSIQ/SPAQ 权重时失败，GitHub release 临时 URL 过期。
由于脚本在错误处退出，后续 vc2_dpo 生成阶段根本没有被执行。

所以 vc2_dpo raw mp4 = 0/4730 的直接原因不是 DPO checkpoint 坏，也不是 GPU 不够，
而是 vc2_base eval 阶段先失败，截断了整个 sequential pipeline。
```

补充注意：

```text
审查里 vc2_base named mp4 = 0/4730 需要进一步确认。
日志里 evaluate.py 的 videos_path 指向 vc2_base/vbench_standard_named，说明该目录曾被用作 VBench 输入。
如果 find -type f 计数为 0，可能是 named 输出为 symlink 或 prepare 阶段未实际落盘。
后续应使用 find -L 或 ls 直接确认。
```

### 21.3 fullmask DiffuEraser run 当前状态

审查输出：

```text
diffueraser_base mp4 = 4720 / 4730
diffueraser_dpo mp4  = 4720 / 4730
```

日志显示 DPO 侧实际跑到了：

```text
[fullmask-vbench-gen] sample=4 prompt=946/946 seed=20230215
[fullmask-vbench-gen] done: .../diffueraser_dpo/vbench_standard_named
```

因此 4720/4730 需要谨慎解释：

```text
可能不是少生成 10 条，而是 VBench prompt 中存在重复 prompt 文本，标准命名 <prompt>-<sample>.mp4 发生覆盖。
也可能确实有 10 个 prompt 因文件名冲突/清洗后冲突而无法区分。
后续需要用 manifest 或 index-preserving filename 解决。
```

fullmask base eval 失败：

```text
subprocess.CalledProcessError:
Command '['wget',
'https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/grit_b_densecap_objectdet.pth',
'-P', '/root/.cache/vbench/grit_model']'
returned non-zero exit status 4.
```

fullmask DPO eval 失败：

```text
RuntimeError:
The server socket has failed to listen on any local network address.
The server socket has failed to bind to [::]:29500 (errno: 98 - Address already in use).
The server socket has failed to bind to 0.0.0.0:29500 (errno: 98 - Address already in use).
```

原因判断：

```text
fullmask 生成基本完成，但 VBench eval 没有成功。
base 侧卡在 VBench 依赖模型下载。
DPO 侧卡在 distributed init 端口冲突，两个 eval 同时/相邻启动时都尝试占用 MASTER_PORT=29500。
```

### 21.4 四个 VBench 表当前状态

截至本次审查：

```text
ordinary VideoDPO VC2 base vs DPO:
  没有成功 VBench 表。
  base 生成完成，但 base eval 失败，DPO 生成没有开始。

fullmask DiffuEraser baseline vs DPO:
  没有成功 VBench 表。
  base/DPO 生成基本完成，但 eval 失败。
```

因此：

```text
四个模型/分支都还没有可用的正式 VBench summary/table。
不能把 raw mp4 数量当作 VBench 结果。
正式结果必须看到 summary.json / summary.csv / vbench_paper_table.md。
```

### 21.5 关于 VBench 是否必须生成几千个视频

正式 full VBench 复现通常需要完整 prompt set：

```text
prompt count = 946
samples_per_prompt = 5
videos_per_model = 946 * 5 = 4730
```

所以对比两个模型时：

```text
2 models * 4730 = 9460 videos
```

ordinary VideoDPO VC2 base vs DPO 需要 9460 个视频。
fullmask DiffuEraser base vs DPO 也需要 9460 个视频。

random30 只适合 qualitative smoke：

```text
30 prompts * 1 sample * 2 models = 60 videos
```

random30 不能替代正式 VBench quantitative 表。

### 21.6 定性 side-by-side 当前状态判断

当前日志只能确认：

```text
fullmask base/DPO 已经生成大量 vbench_standard_named 视频。
```

但不能仅凭本次审查确认：

```text
fullmask qualitative side-by-side folder 已经生成。
```

需要检查：

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/fullmask_qual_smoke/<run>/qual_side_by_side/*.mp4
```

ordinary VideoDPO VC2 的 qualitative side-by-side 尚不能认为完成，因为本次 full run 中：

```text
vc2_dpo raw mp4 = 0/4730
```

没有 DPO 视频就无法做 vc2_base vs vc2_dpo 二合一。

### 21.7 修复方向

不要重跑所有生成。优先补救：

```text
1. 先准备 VBench cache，避免 evaluate.py 临时 wget。
2. ordinary VideoDPO:
   - 保留已经完成的 vc2_base raw mp4。
   - 补跑 vc2_dpo inference。
   - 然后分别 eval vc2_base 和 vc2_dpo。
3. fullmask DiffuEraser:
   - 保留已生成的 vbench_standard_named 视频。
   - 修复/预下载 VBench 模型后只重跑 eval。
   - base 和 DPO eval 串行跑，或设置不同 MASTER_PORT，避免 29500 冲突。
4. 最后分别生成两个 vbench_paper_table.md。
5. qualitative side-by-side 单独用 random30 或已有 matched videos 生成，不要混入 formal full VBench。
```

### 21.8 后续应补的工程改动

为了避免再次浪费长跑时间，建议后续修改 wrapper：

```text
1. 将 inference 和 VBench eval 解耦：
   RUN_INFERENCE=1 RUN_VBENCH=0
   RUN_INFERENCE=0 RUN_VBENCH=1
   这样 base eval 失败不会阻止 DPO inference。

2. VBench eval 前做 dependency health-check：
   - 检查 /root/.cache/vbench/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth
   - 检查 /root/.cache/vbench/grit_model/grit_b_densecap_objectdet.pth
   - 缺失时提前失败，不要等 10 小时生成后才失败。

3. VBench eval 设置 MASTER_PORT：
   - ordinary 可用 29510
   - fullmask base 可用 29520
   - fullmask dpo 可用 29521

4. fullmask filename 增加 index 前缀：
   当前 <prompt>-<sample>.mp4 可能覆盖重复 prompt。
   建议输出类似：
     000001__<sanitized_prompt>-<sample>.mp4
   同时保持 VBench prompt 映射 manifest。
```

## 22. 2026-05-19 续写：VC2/FullMask 定性视频完成、VBench 依赖全量修复与正式 eval 重启

本节记录 2026-05-19 在 PAI 上继续修复 VBench formal evaluation 的完整过程。

本节的核心结论：

```text
1. VideoDPO / VC2 的 DPO 视频已经补跑完成。
2. VideoDPO / VC2 的二合一定性视频已经完成 random30。
3. FullMask / DiffuEraser 的二合一定性视频已经完成 random30。
4. 当前主要问题已经从“视频生成”转为“VBench 定量评测环境依赖”。
5. VBench 16 个维度 import smoke 已经最终通过 BAD_LIST=[]。
6. 已经重新启动 VC2 formal VBench eval。
```

### 22.1 当前实验状态总览

ordinary VideoDPO / VC2：

```text
VC2_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_full/pai_vc2_vs_dpo_full_20260517_034228

vc2_base raw mp4:
  4730 / 4730

vc2_dpo raw mp4:
  4730 / 4730

vc2_dpo raw sample_0:
  946 / 946
vc2_dpo raw sample_1:
  946 / 946
vc2_dpo raw sample_2:
  946 / 946
vc2_dpo raw sample_3:
  946 / 946
vc2_dpo raw sample_4:
  946 / 946

vc2_base/vc2_dpo named count:
  4720
```

解释：

```text
raw mp4 = 4730/4730 才是生成完整性的主判断。
named mp4 = 4720 很可能来自 VBench prompt 文本重复或文件名清洗后冲突覆盖。
prepare-vbench 日志对每个 sample 都显示 written=946 missing=0，因此不应把 4720 简单理解为生成失败。
```

FullMask / DiffuEraser：

```text
FM_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai_fullmask_diffueraser_vs_dpo_full_20260517_035156

diffueraser_base:
  已有 vbench_standard_named 视频。

diffueraser_dpo:
  已跑到 sample=4 prompt=946/946 并显示 done。

定性 side-by-side:
  fullmask random30 二合一定性视频已经生成成功，输出数量 30。
```

定性结果：

```text
VideoDPO / VC2:
  base vs dpo 二合一 random30 已完成。

FullMask / DiffuEraser:
  base vs dpo 二合一 random30 已完成。

这两类定性视频已经可以用于人工观察和展示。
但它们不是 VBench quantitative score/table。
```

### 22.2 当前 VBench eval 的进展

截至 2026-05-19 11:15 CST，VC2 formal VBench eval 已经重新启动：

```text
PID=2356407
LOG=/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_full/pai_vc2_vs_dpo_full_20260517_034228/vbench_eval_retry_all_deps_nohup_20260519_111524.log
```

启动前的全量 import smoke 已通过：

```text
[OK] subject_consistency
[OK] background_consistency
[OK] aesthetic_quality
[OK] imaging_quality
[OK] object_class
[OK] multiple_objects
[OK] color
[OK] spatial_relationship
[OK] scene
[OK] temporal_style
[OK] overall_consistency
[OK] human_action
[OK] temporal_flickering
[OK] motion_smoothness
[OK] dynamic_degree
[OK] appearance_style
BAD_LIST= []
```

因此当前不应再做依赖修复或重复启动第二个 VC2 eval。正确动作是监控该日志，等待 `vc2_base` 和 `vc2_dpo` 的 eval 完成，并检查是否生成：

```text
summary.json
summary.csv
vbench_paper_table.md
vbench_paper_table.csv
vbench_paper_table.tex
```

### 22.3 2026-05-19 VBench 依赖修复链路

本次 formal eval 卡住并不是模型生成问题，而是 VBench 依赖环境连续缺项。完整链路如下。

#### 22.3.1 OpenAI CLIP 包安装位置错误

最初执行：

```bash
"$VIDEODPO_ENV/bin/pip" install /mnt/nas/hj/env_packs/openai_clip_main.zip
```

日志显示虽然安装成功，但实际装到了 `/usr/local/lib/python3.10/site-packages`，导致：

```text
ModuleNotFoundError: No module named 'clip'
```

正确方式：

```bash
VIDEODPO_ENV=/mnt/nas/hj/conda_envs/videodpo
"$VIDEODPO_ENV/bin/python" -m pip install ftfy regex tqdm -i https://mirrors.aliyun.com/pypi/simple/
"$VIDEODPO_ENV/bin/python" -m pip install --no-deps /mnt/nas/hj/env_packs/openai_clip_main.zip
```

验证成功：

```text
clip = /mnt/nas/hj/conda_envs/videodpo/lib/python3.10/site-packages/clip/__init__.py
```

经验：

```text
在 conda/venv 混杂的 PAI 环境中，不要信任 "$ENV/bin/pip"。
优先使用 "$ENV/bin/python" -m pip。
```

#### 22.3.2 `clip` 与新版 setuptools 不兼容

OpenAI CLIP 老代码中有：

```python
from pkg_resources import packaging
```

新版 setuptools 中该导入会失败：

```text
ImportError: cannot import name 'packaging' from 'pkg_resources'
```

已修复：

```bash
CLIP_FILE=/mnt/nas/hj/conda_envs/videodpo/lib/python3.10/site-packages/clip/clip.py
sed -i 's/from pkg_resources import packaging/import packaging.version/' "$CLIP_FILE"
```

修复后关键行：

```text
6:import packaging.version
23:if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
223:    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
```

#### 22.3.3 `pyiqa` 缺失

VBench `imaging_quality.py` 需要：

```python
from pyiqa.archs.musiq_arch import MUSIQ
```

缺失时报错：

```text
ModuleNotFoundError: No module named 'pyiqa'
NotImplementedError: UnImplemented dimension imaging_quality!, No module named 'pyiqa'
```

已安装：

```bash
VIDEODPO_ENV=/mnt/nas/hj/conda_envs/videodpo
"$VIDEODPO_ENV/bin/python" -m pip install "pyiqa==0.1.8" -i https://mirrors.aliyun.com/pypi/simple/
```

安装后：

```text
pyiqa OK
MUSIQ OK
imaging_quality OK
```

#### 22.3.4 ViCLIP 内部也有 packaging 老导入

VBench 的 ViCLIP 文本模块同样有老导入：

```python
from pkg_resources import packaging
```

报错影响：

```text
temporal_style
overall_consistency
```

已修复：

```bash
VICLIP_TEXT=/mnt/nas/hj/H20_Video_inpainting_DPO/external/VBench/vbench/third_party/ViCLIP/viclip_text.py
sed -i 's/from pkg_resources import packaging/import packaging.version/' "$VICLIP_TEXT"
```

修复后：

```text
[OK] temporal_style
[OK] overall_consistency
```

#### 22.3.5 `easydict` 缺失

VBench `dynamic_degree.py` 需要：

```python
from easydict import EasyDict as edict
```

缺失时报错：

```text
ModuleNotFoundError: No module named 'easydict'
```

已安装：

```bash
"$VIDEODPO_ENV/bin/python" -m pip install \
  easydict packaging ninja fvcore iopath pycocotools omegaconf cloudpickle termcolor tabulate yacs portalocker \
  -i https://mirrors.aliyun.com/pypi/simple/
```

修复后：

```text
[OK] dynamic_degree
```

#### 22.3.6 `detectron2` 缺失

以下四个 VBench 维度依赖 GRIT/DenseCaptioning，进而依赖 Detectron2：

```text
object_class
multiple_objects
color
spatial_relationship
```

缺失时报错：

```text
ModuleNotFoundError: No module named 'detectron2'
```

Detectron2 编译前，PyTorch 2.1.2 的 `torch.utils.cpp_extension` 也会因为 setuptools 太新失败：

```text
from pkg_resources import packaging
ImportError: cannot import name 'packaging' from 'pkg_resources'
```

解决方式是把 `videodpo` 环境里的 setuptools 降到 70 以下：

```bash
VIDEODPO_ENV=/mnt/nas/hj/conda_envs/videodpo
"$VIDEODPO_ENV/bin/python" -m pip install "setuptools<70" "wheel" -i https://mirrors.aliyun.com/pypi/simple/
```

验证：

```bash
"$VIDEODPO_ENV/bin/python" - <<'PY'
import setuptools, pkg_resources
from pkg_resources import packaging
import torch
import torch.utils.cpp_extension as cpp
print("setuptools =", setuptools.__version__)
print("pkg_resources.packaging OK:", packaging.version.parse("1.0"))
print("torch =", torch.__version__)
print("cpp_extension OK:", cpp.__file__)
PY
```

Detectron2 build/install 最终成功：

```text
torch = 2.1.2+cu121
cuda = 12.1
cuda_available = True
capability = (9, 0)
TORCH_CUDA_ARCH_LIST=9.0

Successfully built detectron2
Successfully installed detectron2-0.6
detectron2_install_status=0
detectron2 OK: /mnt/nas/hj/conda_envs/videodpo/lib/python3.10/site-packages/detectron2/__init__.py
```

#### 22.3.7 `lvis` 缺失

Detectron2 安装后，GRIT 继续暴露 `lvis` 缺失：

```text
ModuleNotFoundError: No module named 'lvis'
```

影响维度：

```text
object_class
multiple_objects
color
spatial_relationship
```

已安装：

```bash
"$VIDEODPO_ENV/bin/python" -m pip install lvis -i https://mirrors.aliyun.com/pypi/simple/
```

最终全量 import smoke 通过：

```text
BAD_LIST= []
```

### 22.4 VBench cache 已补齐的资产

从 H20 下载并同步到 PAI 的 VBench cache 包括：

```text
/root/.cache/vbench/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth
/root/.cache/vbench/grit_model/grit_b_densecap_objectdet.pth
/root/.cache/vbench/caption_model/tag2text_swin_14m.pth
/root/.cache/vbench/ViCLIP/ViClip-InternVid-10M-FLT.pth
/root/.cache/vbench/ViCLIP/bpe_simple_vocab_16e6.txt.gz
/root/.cache/vbench/umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth
/root/.cache/vbench/amt_model/amt-s.pth
/root/.cache/vbench/raft_model/models/raft-things.pth
```

CLIP cache 包括：

```text
/root/.cache/clip/ViT-B-32.pt
/root/.cache/clip/ViT-L-14.pt
/root/.cache/vbench/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth
```

注意：

```text
VBench 的 init_submodules smoke 只能证明 cache 权重基本可初始化，
不能证明所有 dimension module 都能 import。
必须再跑 16 维 import smoke。
```

### 22.5 为什么终端会“闪退”

多次出现 VS Code / PAI terminal 提示：

```text
The terminal process "/bin/bash" terminated with exit code: 1.
The terminal process "/bin/bash" terminated with exit code: 255.
```

常见原因：

```text
1. 在交互式终端里执行 set -euo pipefail。
2. 后续任何命令返回非 0，主 /bin/bash 直接退出。
3. PAI/VS Code 终端面板只显示 exit code，看不到前面的真实错误。
4. 使用 PYTHONPATH="$PWD/external/VBench:$PYTHONPATH" 时，如果 PYTHONPATH 未定义且 set -u 打开，会立刻 unbound variable 退出。
5. 长命令粘贴过程中被截断，脚本内容不完整，也可能退出。
```

后续原则：

```text
不要在交互式终端直接使用 set -euo pipefail。
先执行：
  set +e
  set +u
  set +o pipefail 2>/dev/null || true

需要 strict mode 时，把命令写入 /tmp/*.sh，再用 bash script.sh > log 2>&1 跑。
长任务必须使用 nohup + LOG。
涉及 PYTHONPATH 时用：
  PYTHONPATH="$PWD/external/VBench:${PYTHONPATH:-}"
```

### 22.6 当前正式 VC2 eval 的监控命令

当前 VC2 eval 日志：

```bash
VC2_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_full/pai_vc2_vs_dpo_full_20260517_034228
LOG=/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_full/pai_vc2_vs_dpo_full_20260517_034228/vbench_eval_retry_all_deps_nohup_20260519_111524.log
```

监控：

```bash
cd /mnt/nas/hj/H20_Video_inpainting_DPO || exit 1

VC2_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_full/pai_vc2_vs_dpo_full_20260517_034228
LOG=/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_full/pai_vc2_vs_dpo_full_20260517_034228/vbench_eval_retry_all_deps_nohup_20260519_111524.log

echo "LOG=$LOG"
pgrep -af 'run_vc2_eval_retry_clip|sc_videodpo_vc2_vbench|evaluate.py|lingbotworldphy' || true
nvidia-smi

echo
echo "========== output files =========="
find "$VC2_ROOT" -maxdepth 3 -type f \( \
  -name 'summary.json' -o \
  -name 'summary.csv' -o \
  -name 'vbench_paper_table.md' -o \
  -name 'vbench_paper_table.csv' -o \
  -name 'vbench_paper_table.tex' -o \
  -name 'results_*_eval_results.json' \
\) -print -exec ls -lh {} \;

echo
echo "========== log tail =========="
tail -220 "$LOG"

echo
echo "========== suspicious =========="
grep -nEi 'Traceback|RuntimeError|Error|Exception|Killed|CUDA out of memory|No such file|ModuleNotFound|ImportError|CalledProcessError|returned non-zero|failed|Address already in use|cannot' "$LOG" | tail -80 || true
```

判断：

```text
还有 evaluate.py 或 lingbotworldphy 进程：
  eval 正在跑。

没有进程且出现 vbench_paper_table.*：
  VC2 eval 正常结束。

没有进程但 log 有 Traceback:
  按最后一个 Traceback 继续修。
```

### 22.7 VC2 eval 成功后要做什么

VC2 eval 成功后不要马上重跑生成。下一步：

```text
1. 检查 vc2_base/vbench_eval 和 vc2_dpo/vbench_eval 里是否有结果文件。
2. 检查根目录是否有 vbench_paper_table.md/csv/tex。
3. 保存/记录最终表格路径和关键分数。
4. 再进入 FullMask / DiffuEraser eval。
```

推荐审查命令：

```bash
cd /mnt/nas/hj/H20_Video_inpainting_DPO || exit 1
VC2_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_full/pai_vc2_vs_dpo_full_20260517_034228

find "$VC2_ROOT" -maxdepth 4 -type f \( \
  -name 'summary.json' -o \
  -name 'summary.csv' -o \
  -name 'vbench_paper_table.md' -o \
  -name 'vbench_paper_table.csv' -o \
  -name 'vbench_paper_table.tex' -o \
  -name 'results_*_eval_results.json' \
\) -print -exec ls -lh {} \;
```

### 22.8 FullMask eval 后续计划

FullMask 生成已经完成，不要重跑生成。

后续只重跑 eval：

```text
RUN_INFERENCE=0
RUN_VBENCH=1
MAKE_PAPER_TABLE=1
```

注意事项：

```text
1. fullmask base 和 dpo eval 必须串行跑，或使用不同 MASTER_PORT。
2. 不要再使用默认 29500 同时跑两个 eval。
3. 复用已经修好的 /mnt/nas/hj/conda_envs/videodpo VBench 环境。
4. 复用已下载的 /root/.cache/vbench 和 /root/.cache/clip。
5. 若 fullmask wrapper 内部同时拉起 base/dpo eval，需要明确设置 MASTER_PORT 或改成顺序执行。
```

推荐端口规划：

```text
VC2 eval:
  29571 或当前脚本内端口。

FullMask base eval:
  29581

FullMask dpo eval:
  29582
```

### 22.9 当前不要做的事

```text
1. 不要重跑 vc2_base inference。
2. 不要重跑 vc2_dpo inference。
3. 不要重跑 fullmask base/dpo generation。
4. 不要同时启动多个 VBench eval 争抢同一个 MASTER_PORT。
5. 不要在 interactive terminal 里直接 set -euo pipefail。
6. 不要再用 "$VIDEODPO_ENV/bin/pip" 安装关键依赖，统一用 "$VIDEODPO_ENV/bin/python" -m pip。
7. 不要把 random30 二合一定性视频当作 VBench 定量表。
```

### 22.10 当前最短下一步 checklist

```text
[x] VC2 DPO full inference 补跑完成。
[x] VC2 random30 side-by-side 完成。
[x] FullMask random30 side-by-side 完成。
[x] VBench cache 权重补齐。
[x] clip 安装到 videodpo env。
[x] clip packaging patch。
[x] pyiqa 安装。
[x] ViCLIP packaging patch。
[x] easydict / fvcore / iopath / pycocotools 等轻依赖安装。
[x] setuptools 降级到兼容 torch cpp_extension。
[x] detectron2 编译安装成功。
[x] lvis 安装。
[x] VBench 16 维 import smoke BAD_LIST=[]。
[ ] VC2 base/dpo formal VBench eval 完成。
[ ] VC2 vbench_paper_table.* 生成并审查。
[ ] FullMask base/dpo formal VBench eval 完成。
[ ] FullMask vbench_paper_table.* 生成并审查。
```

## 23. 2026-05-20 FullMask-DiffuEraser Stage2 修正计划

### 23.1 现象和结论

VC2-base / VC2-DPO 的 VBench side-by-side 视频正常，但 DiffuEraser-base / FullMask-DPO 曾出现：

```text
1. DiffuEraser-base 近乎全黑。
2. FullMask-DPO 色彩和结构异常。
3. 修正 mask 后可以生成合理图像，但 16 帧几乎静止。
```

这不是 VBench 的问题。VBench 依赖已经打通，后续可以最后再测。

根因分两层：

```text
mask 极性错误:
  DiffuEraser pipeline 接收 PIL mask 后会做内部转换：
    original_mask = (original_mask.sum(1)[:, None, :, :] < 0)
  因此 white PIL mask -> internal mask channel 0 -> hole / 需要生成。
  black PIL mask -> internal mask channel 1 -> known / 保留输入。

  训练 dataset 里的 VIDEODPO_FULL_MASK_VALUE=0.0 是正确的 internal full-hole 语义。
  生成脚本以前把 FULL_MASK_VALUE=0.0 直接做成 black PIL mask，等价于 keep all-black input，
  所以 base 会全黑。

运动缺失:
  当前 fullmask Stage1 只训练和生成 UNet2D + BrushNet。
  tools/generate_diffueraser_fullmask_vbench.py 以前遇到 UNetMotionModel 还会 extract_2d_from_motion，
  直接丢掉 temporal module。
  Stage1 pipeline 还会把同一个 initial noise repeat 到所有 frames。
  所以 Stage1 可以学到 prompt 图像生成，但不能保证视频运动。
```

### 23.2 DiffuEraser 论文里的训练设定

DiffuEraser 本身是两段训练：

```text
Stage1:
  训练 BrushNet + main denoising UNet，不带 motion module。
  目标是增强内容生成能力。

Stage2:
  在 main denoising UNet 中训练 motion module。
  冻结 VAE / text encoder / UNet2D / BrushNet。
  目标是增强 temporal consistency 和帧间运动。
```

因此当前已经训练好的 fullmask Stage1 可以继续作为 Stage2 初始化：

```text
policy:
  baseline DiffuEraser UNetMotionModel 提供 temporal module。
  当前 fullmask Stage1 的 unet_main 覆盖 2D 权重。
  当前 fullmask Stage1 的 brushnet 继续使用并冻结。
  只训练 motion module。

ref:
  使用 baseline DiffuEraser converted_weights_step48000 的完整 UNetMotionModel + BrushNet，冻结。
```

### 23.3 新的代码约定

生成脚本约定：

```text
--full_mask_value 默认表示 internal BrushNet mask channel。
internal 0.0 = full hole / 需要生成。
internal 1.0 = known / 保留输入。

生成 PIL mask 时必须反转：
  pil_mask_pixel = round((1.0 - internal_mask_value) * 255)

如果必须复现实验中旧的 PIL 语义，才使用：
  --mask_value_space pil
```

训练脚本约定：

```text
VIDEODPO_FULL_MASK_VALUE=0.0
不要改成 1.0。
训练 dataset 直接输出 internal mask channel，不经过 PIL mask 预处理。
```

Stage2 fullmask setting 必须和当前 Stage1 fullmask setting 对齐：

```text
DPO_DATASET_TYPE=videodpo_fullmask
VIDEODPO_FULL_MASK_VALUE=0.0
TRAIN_HEIGHT=320
TRAIN_WIDTH=512
NFRAMES=16
MAX_EPOCHS=5  # matches completed Stage1 run 20260516_024051
BETA_DPO=5000
LR=6e-6
NUM_GPUS=4
GRAD_ACCUM=4
LOGGING_STEPS=499
NUM_WORKERS=16
MIXED_PRECISION=no
SPLIT_POS_NEG_FORWARD=0
```

训练进程名统一：

```text
WORLDMODELPHY_PROCESS_NAME=lingbotworld-phy
PROCESS_TITLE=lingbotworld-phy
```

### 23.4 PAI Stage2 smoke

在 PAI 机器上先 pull 最新代码，然后跑 1 GPU / 1 step smoke：

```bash
set +e
set +u
set +o pipefail 2>/dev/null || true

cd /mnt/nas/hj/H20_Video_inpainting_DPO || exit 1
git pull --ff-only

STAGE1_WEIGHTS=/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260516_024051_lingbot-world-model-fullmask-videodpo-epoch5-gpu4-7-20260516_104046/last_weights
VC2_YAML=/mnt/nas/hj/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.pai.yaml
LOG="$PWD/logs/fullmask_stage2_smoke_$(date +%Y%m%d_%H%M%S).log"

CUDA_VISIBLE_DEVICES=7 \
NUM_GPUS=1 \
GRAD_ACCUM=4 \
CONDA_ENV=/mnt/nas/hj/conda_envs/diffueraser \
WEIGHTS_DIR=/mnt/nas/hj/weights \
PRETRAINED_DPO_S1="$STAGE1_WEIGHTS" \
STAGE1_WEIGHTS_PATH="$STAGE1_WEIGHTS" \
BASELINE_UNET_PATH=/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000 \
REF_MODEL_PATH=/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000 \
DPO_DATA_ROOT="$VC2_YAML" \
REPORT_TO=none \
MAX_STEPS=1 \
MAX_EPOCHS= \
CKPT_STEPS=1 \
VAL_STEPS=999999 \
LOGGING_STEPS=1 \
RUN_NAME=pai-fullmask-stage2-smoke-$(date +%Y%m%d_%H%M%S) \
WORLDMODELPHY_PROCESS_NAME=lingbotworld-phy \
PROCESS_TITLE=lingbotworld-phy \
NCCL_DEBUG=WARN \
TORCH_DISTRIBUTED_DEBUG=OFF \
TMPDIR=/tmp/hj_worldmodel_tmp \
bash DPO_finetune/scripts/sc_videodpo_fullmask_diffueraser_stage2.sbatch 2>&1 | tee "$LOG"

echo "LOG=$LOG"
```

监控：

```bash
pgrep -af 'lingbotworld-phy|train_stage2.py|run_stage2.py|accelerate' || true
nvidia-smi
tail -220 "$LOG"
grep -nEi 'Traceback|RuntimeError|Error|Exception|Killed|CUDA out of memory|No such file|ModuleNotFound|ImportError|failed|TypeError' "$LOG" | tail -80 || true
```

smoke 通过后再跑正式 Stage2：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
NUM_GPUS=4 \
GRAD_ACCUM=4 \
CONDA_ENV=/mnt/nas/hj/conda_envs/diffueraser \
WEIGHTS_DIR=/mnt/nas/hj/weights \
PRETRAINED_DPO_S1="$STAGE1_WEIGHTS" \
BASELINE_UNET_PATH=/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000 \
REF_MODEL_PATH=/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000 \
DPO_DATA_ROOT="$VC2_YAML" \
REPORT_TO=wandb \
MAX_STEPS= \
MAX_EPOCHS=5 \
LOGGING_STEPS=499 \
NUM_WORKERS=16 \
NCCL_DEBUG=WARN \
TORCH_DISTRIBUTED_DEBUG=OFF \
TMPDIR=/tmp/hj_worldmodel_tmp \
RUN_NAME=pai-fullmask-stage2-epoch5-gpu4-7-$(date +%Y%m%d_%H%M%S) \
WORLDMODELPHY_PROCESS_NAME=lingbotworld-phy \
PROCESS_TITLE=lingbotworld-phy \
nohup bash DPO_finetune/scripts/sc_videodpo_fullmask_diffueraser_stage2.sbatch \
  > logs/fullmask_stage2_train_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
```

### 23.5 Stage2 后的生成测试

Stage2 训练产物会是 `last_weights/unet_main` 中的 `UNetMotionModel`。生成端必须走 Stage2 pipeline，不能再拆成 2D。

```bash
STAGE2_WEIGHTS=/path/to/stage2/run/last_weights
PAIR_ROOT="$PWD/logs/fullmask_stage2_gen_smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PAIR_ROOT"
head -5 external/VideoDPO/prompts/vbench_standard_prompts.txt > "$PAIR_ROOT/prompts5.txt"

CUDA_VISIBLE_DEVICES=0 \
CONDA_ENV=/mnt/nas/hj/conda_envs/diffueraser \
BASE_MODEL_PATH=/mnt/nas/hj/weights/stable-diffusion-v1-5 \
VAE_PATH=/mnt/nas/hj/weights/sd-vae-ft-mse \
OUT_ROOT="$PAIR_ROOT/fullmask_stage2" \
WEIGHTS_PATH="$STAGE2_WEIGHTS" \
FULL_MASK_VALUE=0.0 \
FULL_MASK_VALUE_SPACE=internal \
GEN_STAGE=auto \
GUIDANCE_SCALE=2.0 \
PROMPT_LIMIT=5 \
SAMPLES_PER_PROMPT=1 \
GENERATE=1 \
RUN_VBENCH=0 \
bash DPO_finetune/scripts/h20_diffueraser_fullmask_vbench.sh 2>&1 | tee "$PAIR_ROOT/gen.log"
```

检查运动：

```bash
python - <<'PY'
import cv2, glob, numpy as np
for p in sorted(glob.glob("logs/fullmask_stage2_gen_smoke_*/*/vbench_standard_named/*.mp4")):
    cap = cv2.VideoCapture(p)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f.astype(np.float32))
    cap.release()
    if len(frames) < 2:
        print("[BAD]", p, "frames=", len(frames))
        continue
    diffs = [np.mean(np.abs(frames[i] - frames[i-1])) for i in range(1, len(frames))]
    print(f"{np.mean(diffs):8.3f}  {np.max(diffs):8.3f}  {len(frames):3d}  {p}")
PY
```

如果 Stage2 后仍几乎静止，再排查：

```text
1. 生成脚本是否 resolved_stage=stage2。
2. unet_main/config.json 的 _class_name 是否为 UNetMotionModel。
3. Stage2 训练日志里的 Trainable params 是否只包含 MotionModule。
4. FULL_MASK_VALUE 是否仍是 0.0 internal。
5. 是否误传 GEN_STAGE=stage1 或 --mask_value_space pil。
```
