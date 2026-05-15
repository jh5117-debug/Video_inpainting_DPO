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
