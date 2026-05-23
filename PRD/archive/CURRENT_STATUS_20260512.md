# Current Status: 2026-05-12

本文件记录 2026-05-12 的最新项目状态，给新的聊天框或新的 Codex 终端快速接手使用。它应优先于旧的历史报告阅读，但仍需要结合：

- `PRD/README_FOR_NEXT_CHAT.md`
- `PRD/NEXT_CHAT_FULL_CONTEXT_20260509.md`
- `PRD/PROJECT_HANDOFF_20260509.md`
- `PRD/meeting_followup_videodpo_repro_and_bridge_20260511.md`

## 1. 当前最高优先级任务

导师最新要求分成两条线：

1. **Task 1: 完整复现 VideoDPO/VC2，并用 VBench 对齐论文指标。**
   之前的 DPO metric panel 只能说明训练行为，不能证明 VideoDPO 复现完成。必须用 VBench prompt suite 生成视频，并计算 VideoDPO 论文报告的 VBench 相关分数。

2. **Task 2: 从已知可工作的 VideoDPO 出发，最小变量迁移到 DiffuEraser。**
   数据和 task 暂时保持 VideoDPO，不先改成 inpainting；只把模型换成 DiffuEraser full-mask 形式。full-mask 的意思是把 inpainting 模型的 mask 设成全图，让它在没有可见区域的情况下按 prompt 生成整段视频。

## 2. 三台服务器工作流

### HAL

路径：

```text
/home/hj/Video_inpainting_DPO
```

HAL 是开发源头：读代码、改代码、写 PRD、做本地 smoke、commit/push。不要在 H20 或 SC 上临时改训练代码后忘记同步。

当前 HAL 工作区有用户已有删除状态，接手者不要随手 revert：

```text
PRD/First_Finetuning_Summary.md
PRD/PPT.pptx
PRD/stage2_motion_module_init.md
PRD/update_ppt_tables.py
PRD/validation_optimization.md
```

这些不是本次文档更新要处理的内容。

### H20

H20 是 GPU 训练机，使用 bash launcher，不使用 Slurm。不要把 SC 的 `sbatch` 命令给 H20。

### SC

SC 是合作者 Slurm 训练机。必须保留环境变量路径逻辑，例如：

```text
PROJECT_HOME
PROJECT_DEV
PROJECT_DATA
DATA_NAME
DATA
DPO_DATA_ROOT
VAL_DATA_DIR
WEIGHTS_DIR
EXPERIMENTS_DIR
WANDB_PROJECT
WANDB_ENTITY
```

不要把 `/home/nvme01/...` 或 HAL 的 `/home/hj/...` 写进 SC 脚本。

## 3. SC VideoDPO/VC2 当前状态

当前 SC 工作流已经改为 repo-local submodules，不再使用 `${PROJECT_DEV}/VideoDPO` 或 `${PROJECT_DEV}/VBench` 这样的 sibling naked clone。

依赖位置：

```text
external/VideoDPO
external/VBench
```

SC 初始化和健康检查：

```bash
source ~/.bashrc
cd "$PROJECT_DEV/Video_inpainting_DPO"
git pull --ff-only origin main
bash DPO_finetune/scripts/sc_videodpo_pull_submodules_and_health_check.sh
```

截至 2026-05-12，用户在 SC 上运行：

```bash
CONDA_ENV=diffueraser bash DPO_finetune/scripts/sc_videodpo_health_check.sh
```

输出为：

```text
repo_commit=65e916f Fix VC2 health clip parsing
========== Summary ==========
errors=0 warnings=0
[RESULT] PASS: static health check found required assets.
```

这说明静态审查已通过：submodules、VC2 dataset yaml、metadata/pair、首个视频路径、VC2 base/ref checkpoints、VBench prompt/config、DiffuEraser bridge 权重、脚本语法都被找到。

### 3.1 metadata 误报已经修复

之前的 `first clip missing` 不是 HF 数据集一定坏了，而是 health check 解析一整行 `metadata.json` 的 bug：

- `metadata.json` 被写成单行 JSON。
- 旧脚本用 `grep -m1 -o clip_path`，会从同一行里抓到多个 `clip_path`。
- Bash 变量里塞入多个路径后，`-f` 判断失败，误报第一个视频不存在。

已修复：

- `sc_videodpo_health_check.sh` 现在只取第一个 `clip_path`。
- 缺文件时只做 bounded/timeout scan，不再长时间卡住。
- `tools/prepare_videodpo_vc2_dataset.py` 会输出 `first_clip_exists=yes/no`，并在首个 clip 不存在时直接失败，避免 prepare 和 health check 互相矛盾。

相关提交：

```text
65e916f Fix VC2 health clip parsing
d7a2e37 Recover bad VC2 metadata rewrites
```

## 4. SC VideoDPO/VC2 训练配置

当前 SC 的 VideoDPO/VC2 训练脚本：

```text
DPO_finetune/scripts/sc_videodpo_vc2_train.sbatch
```

Slurm 资源配置已按用户要求改成：

```bash
#SBATCH --job-name=VDPO_VC2
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/dpo-stage1-%j.out
```

脚本默认训练也同步为 8 卡：

```text
NUM_GPUS=8
DEVICE_LIST=0,1,2,3,4,5,6,7
BATCH_SIZE=1
GRAD_ACCUM=2
CKPT_EVERY=499
BETA_DPO=5000
NUM_WORKERS=16
video_length=16
resolution=[320,512]
```

重要说明：

- 不传 `MAX_OPT_STEPS` 时，仍使用官方 VideoDPO config 里的 `max_epochs=10`。
- 传 `MAX_OPT_STEPS=10000` 会变成固定 optimizer-step 内部对比实验，不再是纯官方 epoch 复现。
- 官方 `external/VideoDPO/configs/vc2_dpo/run.sh` 是 4 GPU；当前 SC 脚本按用户要求改为 8 GPU。因此模型/config 超参接近官方，但 world size 与官方 run.sh 不完全一致。若要严格复刻官方 run.sh，应显式覆盖 `NUM_GPUS=4 DEVICE_LIST=0,1,2,3`，或另开实验记录。
- 官方 README 建议训练时不要提供 validation dataloader 以降低显存峰值；当前训练脚本也不跑 Lightning validation。用户自己的 validation 是训练后 VBench sweep。

当前推荐启动训练命令：

```bash
source ~/.bashrc
cd "$PROJECT_DEV/Video_inpainting_DPO"
git pull --ff-only origin main
CONDA_ENV=diffueraser bash DPO_finetune/scripts/sc_videodpo_health_check.sh

CONDA_ENV=diffueraser \
RUN_NAME=sc-vc2-dpo-official-beta5000 \
BETA_DPO=5000 \
sbatch --export=ALL DPO_finetune/scripts/sc_videodpo_vc2_train.sbatch
```

## 5. W&B 当前配置

`sc_videodpo_vc2_train.sbatch` 已经和原 DiffuEraser DPO stage1/stage2 对齐 W&B：

```text
WANDB_ENTITY=jh5117-columbia-university
WANDB_PROJECT=DPO_Diffueraser
WANDB_RUN_GROUP=VideoDPO_VC2
WANDB_DIR=${PROJECT_ROOT}/.wandb_cache
WANDB_CACHE_DIR=${PROJECT_ROOT}/.wandb_cache
WANDB_DATA_DIR=${PROJECT_ROOT}/.wandb_cache
WANDB_CONFIG_DIR=${PROJECT_ROOT}/.wandb_cache/config
```

如果 `WANDB_API_KEY` 未设置，脚本会尝试从 `${HOME}/.netrc` 读取 `api.wandb.ai` 的 password。缓存写在项目目录，不写 home。

默认启用 Lightning `WandbLogger`。禁用 W&B：

```bash
ENABLE_WANDB=0 sbatch --export=ALL DPO_finetune/scripts/sc_videodpo_vc2_train.sbatch
```

2026-05-12 追加修复：

- SC 第一次 VC2-DPO 训练在 `lvdm/modules/encoders/condition.py` 处失败，原因是复用的 `diffueraser` conda env 缺少 `kornia`。
- `videodpo_env_smoke_and_export.sh` 的 minimal install/smoke 现在包含 `kornia` 和 `wandb`，并会显式 import `lvdm.modules.encoders.condition`。
- `sc_videodpo_health_check.sh` 现在默认做轻量 Python import preflight；它仍然不跑训练、不跑 inference、不占 GPU。
- `sc_videodpo_vc2_train.sbatch` 现在会在 `torchrun` 前创建 W&B launcher marker，并在失败时把 Slurm launch log 的 tail 写入同一个 W&B run，避免 import 阶段报错时 W&B 页面没有 run。

修复 SC 环境的命令：

```bash
source ~/.bashrc
cd "$PROJECT_DEV/Video_inpainting_DPO"
git pull --ff-only origin main
CONDA_ENV=diffueraser bash DPO_finetune/scripts/sc_videodpo_fix_env_and_health_check.sh
```

这个 wrapper 会先 `git submodule update --init --recursive`，再显式用 `INSTALL_MINIMAL=1` 修复/检查当前 conda env，最后跑 `sc_videodpo_health_check.sh`。普通 health check 仍然只检查，不会偷偷改环境。

2026-05-12 追加修复：

- SC 训练曾在 dependency preflight 处报 `KeyError: 'VIDEODPO_REPO'`。直接原因是 `sc_videodpo_vc2_train.sbatch` 里 `VIDEODPO_REPO` 只是 shell 变量，Python preflight 用 `os.environ["VIDEODPO_REPO"]` 读取不到。训练脚本已改为显式 `export VIDEODPO_REPO`，并一起 export 数据、日志、DPO diagnostics 等关键变量。
- 官方 VideoDPO VC2 代码默认只记录有限训练信号，不能满足我们之前分析用的完整 DPO 中间指标。现在新增 repo 内补丁 `patches/videodpo/sc_vc2_dpo_diagnostics.patch`，训练脚本默认 `APPLY_DPO_DIAG_PATCH=1`，启动前会把补丁应用到 `external/VideoDPO/lvdm/models/ddpm3d.py`。如果以后要跑完全原版 VideoDPO，可显式设置 `APPLY_DPO_DIAG_PATCH=0`。
- 新 diagnostics 不改变 VideoDPO 的模型结构、数据集、task，也不改 Slurm 资源；它只补充 DPO loss 内部统计和日志。当前补丁保留官方 `dpo_loss` objective，只把 diagnostics 里的 winner/loser MSE 和 `implicit_acc` 按 **video pair** 聚合，而不是遗留的部分维度聚合。
- W&B 中会出现标量：`global/implicit_acc`、`global/inside_term_mean/min/max`、`global/loser_dominant_ratio`、`rank0/dpo_loss`、`rank0/mse_w`、`rank0/ref_mse_w`、`rank0/mse_l`、`rank0/ref_mse_l`、`rank0/win_gap`、`rank0/lose_gap`、`rank0/reward_margin`、`rank0/sigma_term`、`rank0/kl_divergence`、`train/factor`。
- 每 `DPO_DIAG_EVERY=300` 个 Lightning optimizer `global_step`，rank0 会打印一行 `[dpo_diag] ...`，并向 W&B 更新累计表：`dpo/diagnostics_table`。默认 `DPO_DIAG_EVERY=300`，可在提交 Slurm 时覆盖。
- `sc_videodpo_health_check.sh` 会检查这个 diagnostics patch 是否存在、是否已经应用或能否干净 apply，避免排队到 GPU 后才发现补丁失效。

相关提交：

```text
f8c68d5 Add SC VideoDPO DPO diagnostics
dd9f8be Align SC VideoDPO W&B logging
0a763d5 Use 8 GPUs for SC VideoDPO VC2 training
```

最新 SC 训练命令：

```bash
source ~/.bashrc
cd "$PROJECT_DEV/Video_inpainting_DPO"
git pull --ff-only origin main
CONDA_ENV=diffueraser bash DPO_finetune/scripts/sc_videodpo_fix_env_and_health_check.sh

CONDA_ENV=diffueraser \
RUN_NAME=sc-vc2-dpo-official-beta5000 \
BETA_DPO=5000 \
DPO_DIAG_EVERY=300 \
sbatch --export=ALL DPO_finetune/scripts/sc_videodpo_vc2_train.sbatch
```

## 6. VBench 验证逻辑

用户说的 `val` 不是 Lightning validation loop，而是 VideoDPO 论文口径的 VBench 评估：

```text
VC2 base ckpt
  -> VideoDPO 训练得到 VC2-DPO ckpt
  -> 用 VBench prompts 生成视频
  -> VBench evaluate
  -> 比较 base 与 DPO checkpoint，并和论文 VC2 结果对照
```

训练后 checkpoint sweep：

```bash
CONDA_ENV=diffueraser \
TRAIN_RUN_NAME=sc-vc2-dpo-official-beta5000 \
INCLUDE_LAST_IN_SWEEP=1 \
SELECT_BEST=1 \
sbatch --export=ALL DPO_finetune/scripts/sc_videodpo_vc2_checkpoint_sweep.sbatch
```

输出位置：

```text
${PROJECT_DATA}/experiments/videodpo_vbench_sweeps/<run_name>
${PROJECT_DATA}/experiments/videodpo_vc2_dpo/<run_name>/checkpoints/selected_vbench
```

选择指标默认是 `total_score`，会保留 best checkpoint 和 last checkpoint。不要在确认 sweep 输出前启用 `PRUNE_AFTER=1`。

## 7. DiffDPO 当前诊断口径

DiffDPO stage1/stage2 的 `implicit_acc` 诊断已经改成 **video-pair 粒度**：

- `batch_size=1`、8 卡时，分母是 `8` 个 winner/loser video pair。
- 不再是旧 frame-level 的 `8 * 16 = 128`。
- 当前只改 diagnostics，不改 DPO loss objective。
- DPO loss 仍是 frame-level。

如果未来要把 objective 也改成 pair-level，必须作为新实验单独记录。

## 8. 旧实验结论仍然有效

四个旧日志的核心结论：

1. `普通DiffDPO_loss.log`：裸 DPO 出现 loser-dominant failure，PSNR/SSIM 下降。
2. `把lose_gap删除的loss.log`：winner branch 可稳定工作，PSNR/SSIM 稳定。
3. `VideoDPO的训练.log`：原始 VideoDPO 曲线更平滑，gap 量级小。
4. `使用VideoInpainting的数据集的VideoDPO的loss.log`：换成 inpainting pair 后，VideoDPO 也出现 loser-dominant 倾向。

不要把 `implicit_acc=1` 或 `DPO_loss=0` 当成功。必须结合：

```text
win_gap
lose_gap
mse_w/ref_mse_w
mse_l/ref_mse_l
loser_dominant_ratio
PSNR/SSIM 或 VBench
```

## 9. 接手者安全规则

1. 先读 PRD，再改代码。
2. 先运行 `git status --short`，不要覆盖用户已有改动。
3. 不要 revert 当前工作区里用户已有删除。
4. 不要删除日志、checkpoint、PRD assets。
5. 不要硬编码 SC 路径。
6. 不要把 H20 bash launcher 改成 Slurm。
7. 不要把 SC Slurm 脚本改成 H20 bash。
8. 新训练必须记录 commit、数据路径、GPU 数、batch size、beta、step/epoch 口径、W&B run。
9. 修改 loss 必须写入 PRD，因为它改变实验定义。
10. 修改 diagnostics 必须写清楚不改变训练 objective。
