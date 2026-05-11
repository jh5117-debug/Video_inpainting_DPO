# Video Inpainting DPO 项目交接文档

Date: 2026-05-09

本文件给新的聊天框或新的终端接手项目使用。目标是让接手者完整理解当前项目，不因为缺上下文而损害代码、路径、实验设置或训练服务器。

## 0. 最新阅读入口

如果是新聊天框接手，优先阅读：

```text
PRD/README_FOR_NEXT_CHAT.md
PRD/NEXT_CHAT_FULL_CONTEXT_20260509.md
PRD/PROJECT_HANDOFF_20260509.md
PRD/DPO_Training_Metrics_Explained.md
PRD/dpo_metric_regularization_prd_20260505.md
```

`NEXT_CHAT_FULL_CONTEXT_20260509.md` 是对本文件的更完整操作版，特别强调 HAL/H20/SC 三台机器协作、环境变量路径规则、当前 pair-level diagnostics 代码逻辑、以及给新聊天框的提示词。

## 1. 当前项目一句话总结

这个项目在做 Video Inpainting 的 DPO fine-tuning。当前同时比较两类基底模型：

- DiffuEraser 基底：`DiffDPO` 和 `no-lose-gap ablation`
- VideoDPO 基底：`VideoDPO open-source data` 和 `VideoDPO + VideoInpainting data`

当前最重要的发现是：普通裸 DPO 很容易出现 `implicit_acc` 很高、`DPO_loss` 很低，但真实质量下降的 loser-dominant failure。也就是说模型可能主要通过让 loser 更差来满足偏好排序，而不是让 winner 变好。

## 2. 三台机器的协作逻辑

### 2.1 HAL 服务器，本地开发机

角色：

- HAL 上有 Codex，主要用于开发、读代码、改代码、写 PRD、分析日志。
- 当前本地 repo 路径是 `/home/hj/Video_inpainting_DPO`。
- 开发完成后，应该 commit/push 到 GitHub。

原则：

- HAL 是主要开发源头。
- 修改训练代码、PRD、分析脚本，都先在 HAL 完成。
- 改完至少运行语法检查，例如：

```bash
cd /home/hj/Video_inpainting_DPO
python -m py_compile training/dpo/train_stage1.py training/dpo/train_stage2.py training/dpo/scripts/run_stage1.py
git status --short
```

然后再 push，让训练服务器 pull。

### 2.2 H20 服务器，H20 GPU 训练机

角色：

- H20 上有 H20 GPU，用来跑真实训练。
- 用户通常自己在 H20 终端直接执行 bash 脚本。
- H20 不应该作为主要开发源头，应该 `git pull` 最新代码后跑。

常见路径：

- DiffDPO repo 通常是 `/home/nvme01/Video_inpainting_DPO`
- VideoDPO adapter 项目曾用 `/home/nvme01/H20_Video_inpainting_DPO`
- VideoDPO 原始 repo 曾用 `/home/nvme01/VideoDPO`

H20 运行方式：

- 直接 bash，不走 Slurm。
- 训练前先 `git pull`。
- 训练时通过环境变量覆盖 GPU、beta、step、数据路径。

DiffDPO stage1 beta=10 的 H20 推荐命令：

```bash
source ~/.bashrc
cd /home/nvme01/Video_inpainting_DPO
git pull --ff-only origin main
mkdir -p logs

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NUM_GPUS=8 \
BATCH_SIZE=1 \
BETA_DPO=10 \
DPO_LOSE_GAP_WEIGHT=1.0 \
MAX_STEPS=10000 \
CKPT_STEPS=2000 \
VAL_STEPS=2000 \
RUN_NAME=h20-dpo-stage1-beta10-pair-implicit-acc \
MIXED_PRECISION=bf16 \
SPLIT_POS_NEG_FORWARD=1 \
GRADIENT_CHECKPOINTING=1 \
bash scripts/h20_run_dpo_stage1.sh
```

后台运行版本：

```bash
cd /home/nvme01/Video_inpainting_DPO
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_GPUS=8 BATCH_SIZE=1 \
BETA_DPO=10 DPO_LOSE_GAP_WEIGHT=1.0 MAX_STEPS=10000 CKPT_STEPS=2000 VAL_STEPS=2000 \
RUN_NAME=h20-dpo-stage1-beta10-pair-implicit-acc MIXED_PRECISION=bf16 \
SPLIT_POS_NEG_FORWARD=1 GRADIENT_CHECKPOINTING=1 \
bash scripts/h20_run_dpo_stage1.sh > logs/h20_dpo_stage1_beta10_pairacc.out 2>&1 &
```

查看日志：

```bash
tail -f logs/h20_dpo_stage1_beta10_pairacc.out
```

注意：

- H20 不要使用 SC 的 `sbatch` 脚本。
- 如果 H20 上 repo 不是 `/home/nvme01/Video_inpainting_DPO`，先确认实际路径。
- VideoDPO + Inpainting adapter 使用另一套脚本：`patches/videodpo/apply_and_launch_h20_videodpo.sh`。

### 2.3 SC 服务器，合作者 Slurm 训练机

角色：

- SC 是合作者的训练卡资源。
- 必须使用 Slurm。
- 合作者对路径和环境要求很高，代码里大量依赖环境变量，不能轻易硬编码路径。

SC 原始运行方式：

```bash
source ~/.bashrc
cd "$PROJECT_DEV/Video_inpainting_DPO"
git pull --ff-only origin main
mkdir -p logs
sbatch --export=ALL DPO_finetune/scripts/03_dpo_stage1.sbatch
```

当前 repo 里也有封装脚本：

```bash
bash scripts/sc_submit_dpo_stage1.sh
```

SC 环境变量逻辑：

- `PROJECT_HOME`
- `PROJECT_DEV`
- `PROJECT_DATA`
- `DATA_NAME`
- `DATA`
- `DPO_DATA_ROOT`
- `VAL_DATA_DIR`
- `WEIGHTS_DIR`
- `EXPERIMENTS_DIR`

原则：

- 不要把 SC 路径写死。
- 不要破坏 `--export=ALL` 的变量传递。
- 不要把 H20 的本地路径带到 SC。
- 如果要给 SC 改 beta，可以通过环境变量：

```bash
source ~/.bashrc
cd "$PROJECT_DEV/Video_inpainting_DPO"
git pull --ff-only origin main
mkdir -p logs

BETA_DPO=10 \
MAX_STEPS=10000 \
RUN_NAME=sc-dpo-stage1-beta10-pair-implicit-acc \
sbatch --export=ALL DPO_finetune/scripts/03_dpo_stage1.sbatch
```

如果使用封装脚本：

```bash
BETA_DPO=10 \
MAX_STEPS=10000 \
RUN_NAME=sc-dpo-stage1-beta10-pair-implicit-acc \
bash scripts/sc_submit_dpo_stage1.sh
```

## 3. 当前代码结构和最新改动

### 3.1 当前真正使用的 DPO 训练代码

主要代码在：

- `training/dpo/train_stage1.py`
- `training/dpo/train_stage2.py`
- `training/dpo/scripts/run_stage1.py`
- `scripts/h20_run_dpo_stage1.sh`
- `scripts/h20_run_dpo_stage2.sh`
- `DPO_finetune/scripts/03_dpo_stage1.sbatch`
- `scripts/sc_submit_dpo_stage1.sh`

早期文档里提到 `DPO_finetune/train_dpo_stage1.py`，现在它只是 compatibility wrapper，真正逻辑已经迁移到 `training/dpo/train_stage1.py`。

### 3.2 2026-05-09 最新重要代码改动

用户要求把 DiffDPO 的 `implicit_acc` 生成逻辑改成 video-pair 粒度。

已完成：

- `training/dpo/train_stage1.py`
- `training/dpo/train_stage2.py`

当前逻辑：

1. DPO loss 本身仍保持原来的 frame-level 训练目标。
2. 诊断指标里的 `implicit_acc`、`inside_term_mean/min/max`、`Samples: correct/total`、`loser_dominant_ratio` 改为 video-pair 粒度。
3. 实现方式是先把每个视频的 `nframes` 帧 gap 平均成一个 pair margin，再判断 `inside_term > 0`。
4. 分布式时继续用 `accelerator.gather` 跨卡统计全局 pair-level 指标。

为什么没有改 DPO loss：

- 用户当前明确关注的是 `implicit_acc` 统计粒度。
- 直接改 DPO loss 会改变优化目标，属于更大的实验变更。
- 当前保守做法是只改诊断口径，不改变训练本身。

如果未来要把 DPO loss 也改成 pair-level，必须明确开一个新实验，并在日志和 PRD 里标注，因为它不再和旧 DiffDPO loss 完全可比。

### 3.3 当前 pair-level implicit_acc 定义

对一个 winner/loser video pair：

```text
win_gap  = policy_winner_mse - ref_winner_mse
lose_gap = policy_loser_mse  - ref_loser_mse
inside   = -0.5 * beta * (win_gap - lose_gap_weight * lose_gap)
```

如果：

```text
inside > 0
```

这个 video pair 算判断正确。

直觉：

- policy 相对 reference 更偏向 winner，则正确。
- 普通 DPO 下，大致等价于 `win_gap < lose_gap`。
- 注意这不一定代表 winner 真的变好，因为也可能是 loser 变更差。

### 3.4 batch_size=1, 8 卡时 implicit_acc 怎么算

如果是 DiffDPO 当前新逻辑或 VideoDPO 的 video-pair 逻辑：

```text
batch_size = 1
num_gpus = 8
nframes = 16
```

一次 optimizer step 处理：

```text
8 个 winner/loser video pair
```

因此诊断应类似：

```text
Samples: 4/8 correct
implicit_acc = 0.5
```

不应该再是：

```text
Samples: 64/128 correct
```

后者是旧 frame-level 统计。

## 4. 为什么要改成 video-pair 粒度

DPO 数据的偏好标签是：

```text
这个 winner 视频 比这个 loser 视频好
```

不是：

```text
第 1 帧 winner 比 loser 好
第 2 帧 winner 比 loser 好
...
```

frame-level 问题：

- 一个视频被拆成 16 个判断，会把统计量放大。
- 简单帧、静态帧、mask 小的帧可能主导 `implicit_acc`。
- 很容易出现 `implicit_acc` 过快到 1，但视频级质量并没有真的变好。

video-pair 更合理：

- 先把一个视频片段的所有帧汇总成一个 pair margin。
- 每个 winner/loser 视频对只投一票。
- 更符合偏好数据本身的语义。

## 5. DPO objective 是什么

这里的 objective 指训练时最小化的目标函数，也就是 loss。

普通 Diffusion-DPO：

```text
DPO_loss = -logsigmoid(-0.5 * beta * (win_gap - lose_gap))
```

更完整地说：

```text
win_gap  = policy_winner_mse - ref_winner_mse
lose_gap = policy_loser_mse  - ref_loser_mse
inside   = -0.5 * beta * (win_gap - lose_gap)
DPO_loss = -logsigmoid(inside)
```

危险点：

- 这个 objective 只要求相对排序赢。
- 它可以通过让 winner 更好来赢。
- 也可以通过让 loser 更差来赢。
- 后者就是当前看到的 loser-dominant shortcut。

## 6. 四个实验的当前理解

所有图和 CSV 在：

```text
/home/hj/Video_inpainting_DPO/PRD/assets/dpo_metric_analysis_20260505/
```

主图：

```text
/home/hj/Video_inpainting_DPO/PRD/assets/dpo_metric_analysis_20260505/all_experiments_metric_panels.png
```

详细报告：

```text
/home/hj/Video_inpainting_DPO/PRD/dpo_metric_regularization_prd_20260505.md
```

### 6.1 普通 DiffDPO_loss

基底模型：

- DiffuEraser

日志：

```text
/home/hj/log/普通DiffDPO_loss.log
```

结论：

- `DPO_loss` 很快接近 0。
- `implicit_acc` 很快接近 1。
- 但 `win_gap`、`lose_gap`、MSE 差距暴涨。
- PSNR/SSIM 明显下降。

解释：

- 这是普通裸 DPO 的典型 saturation/collapse。
- 模型主要靠扩大 loser error 来满足排序。
- `implicit_acc=1` 在这里不是好事。

### 6.2 no-lose-gap ablation

基底模型：

- DiffuEraser

日志：

```text
/home/hj/log/把lose_gap删除的loss.log
```

结论：

- `implicit_acc` 高，但 `DPO_loss` 仍在 `0.69` 附近。
- `win_gap` 很小且更稳定。
- PSNR/SSIM 稳定。

注意：

- 这个实验不是单纯删除 lose_gap。
- 它还使用了更小的 beta，并且 negative branch 没有梯度。
- 所以它证明“winner 分支本身不是坏的”，但不能简单等价于标准 DPO。

### 6.3 开源 VideoDPO 原始训练

基底模型：

- VideoDPO

日志：

```text
/home/hj/log/VideoDPO的训练.log
```

结论：

- 指标看起来更正常。
- `implicit_acc` 有增长过程，不是瞬间到 1。
- gap 量级很小，大多在 `1e-3`。

解释：

- VideoDPO 本身按 video pair 统计。
- 它不是 frame-level 拆开判断。
- 所以曲线更平滑、更接近偏好学习真实过程。

### 6.4 VideoDPO + VideoInpainting data

基底模型：

- VideoDPO

日志：

```text
/home/hj/log/使用VideoInpainting的数据集的VideoDPO的loss.log
```

结论：

- 使用 VideoInpainting 数据后，VideoDPO 也出现 loser-dominant 倾向。
- `lose_gap` 后期明显高于 `win_gap`。
- 当前日志曾经缺少 PSNR/SSIM，因为旧环境缺 `skimage`。

注意：

- 后来 H20 脚本里加入过 `INSTALL_SKIMAGE=1` 和 metric preflight。
- 如果新跑，应该确认日志里有：

```text
metric_preflight=ok
```

否则不要声称 PSNR/SSIM 已经成功记录。

## 7. 为什么 VideoDPO 的指标看起来比 DiffDPO 正常

核心原因不是 VideoDPO 一定更好，而是统计粒度不同：

- DiffDPO 旧逻辑按 frame-level 判断。
- VideoDPO 按 video-pair 判断。

比如：

```text
8 GPUs * batch_size 1 * 16 frames = 128 frame-level 判断
```

旧 DiffDPO 的 `implicit_acc` 容易变成 `128/128`。

但 video-pair 逻辑下：

```text
8 GPUs * batch_size 1 = 8 video-pair 判断
```

这更接近偏好标签。

## 8. 为什么 DiffDPO 的 DPO_loss 很快到 0

普通 DiffDPO 曾用较大的 beta，例如 `beta=500`。

只要：

```text
lose_gap > win_gap
```

一点点，乘上 beta 后：

```text
inside = -0.5 * beta * (win_gap - lose_gap)
```

就会很大，`DPO_loss=-logsigmoid(inside)` 会迅速接近 0。

这和 Reg-DPO 论文里的问题一致：

- DPO 只约束 gap，不约束单个样本质量。
- 它可以通过扩大 negative error 降低 loss。
- sigmoid 饱和后梯度会很快变小。
- loss 看起来成功，但视觉质量可能退化。

## 9. 为什么 DiffDPO 的 MSE/gap 比其他实验大很多

不能只说是 DiffuEraser 和 VideoDPO 模型不同。

更关键的是：

- 普通 DiffDPO 的 objective 允许扩大 loser error。
- beta 较大，更容易饱和。
- 旧诊断是 frame-level，更容易虚高。
- DiffuEraser 和 VideoDPO 的 latent/MSE 尺度不同，绝对数值不能直接比。
- Inpainting 任务的偏好差异局部、mask 相关，容易被 loser shortcut 利用。

最强证据：

- no-lose-gap 也是 DiffuEraser 基底。
- 但它的 gap 没有爆，PSNR/SSIM 稳定。
- 所以普通 DiffDPO 的问题主要来自 objective 和训练设置，不是单纯模型架构。

## 10. 当前推荐实验方向

短期：

1. 用 `beta=10` 跑 DiffDPO stage1。
2. 使用新的 pair-level `implicit_acc` 诊断。
3. 保留 PSNR/SSIM validation。
4. 观察 `win_gap` 是否能接近或低于 0。
5. 观察 `lose_gap` 是否不再爆炸。

中期：

1. 加 winner-side SFT/Reg-DPO anchor。
2. 尝试 DPOP/IPO/APO 风格约束。
3. 把 `loser_dominant_ratio` 作为核心 early-warning 指标。
4. 不再只看 `implicit_acc` 和 `DPO_loss`。

不要做：

- 不要看到 `implicit_acc=1` 就认为成功。
- 不要看到 `DPO_loss=0` 就认为成功。
- 不要忽略 PSNR/SSIM。
- 不要把 frame-level 和 video-pair-level 曲线直接混比。

## 11. 当前重要文件清单

代码：

```text
training/dpo/train_stage1.py
training/dpo/train_stage2.py
training/dpo/scripts/run_stage1.py
scripts/h20_run_dpo_stage1.sh
scripts/h20_run_dpo_stage2.sh
scripts/sc_submit_dpo_stage1.sh
DPO_finetune/scripts/03_dpo_stage1.sbatch
patches/videodpo/apply_and_launch_h20_videodpo.sh
patches/videodpo/h20_videoinpaint_dpo_adapter.patch
```

PRD：

```text
PRD/README_FOR_NEXT_CHAT.md
PRD/PROJECT_HANDOFF_20260509.md
PRD/dpo_metric_regularization_prd_20260505.md
PRD/DPO_Training_Metrics_Explained.md
PRD/DPO_Project_Complete_Summary.md
```

实验图和表：

```text
PRD/assets/dpo_metric_analysis_20260505/all_experiments_metric_panels.png
PRD/assets/dpo_metric_analysis_20260505/phase_summary.csv
PRD/assets/dpo_metric_analysis_20260505/delta_summary.csv
PRD/assets/dpo_metric_analysis_20260505/experiment_scope.csv
PRD/assets/dpo_metric_analysis_20260505/all_diagnostics.csv
PRD/assets/dpo_metric_analysis_20260505/all_diagnostics_full.csv
```

日志：

```text
/home/hj/log/普通DiffDPO_loss.log
/home/hj/log/把lose_gap删除的loss.log
/home/hj/log/VideoDPO的训练.log
/home/hj/log/使用VideoInpainting的数据集的VideoDPO的loss.log
```

## 12. 新聊天框接手时的安全规则

必须遵守：

1. 先读 `PRD/README_FOR_NEXT_CHAT.md` 和本文件。
2. 先 `git status --short`，确认工作区已有改动。
3. 不要 revert 用户已有改动。
4. 不要删除日志、checkpoint、PRD assets。
5. 不要硬编码 SC 路径。
6. 不要把 H20 和 SC 的启动方式混用。
7. 修改训练代码后要做语法检查。
8. 如果要改 loss，必须明确写入 PRD，因为这会改变实验定义。
9. 如果只是改 diagnostics，要明确说明不改变训练目标。
10. 新实验要把 beta、step、GPU 数、batch_size、数据路径、代码 commit 都记录下来。

## 13. 接下来最可能的任务

用户接下来可能会要求：

- 把 HAL 当前改动 push 到 GitHub。
- 在 H20 上 pull 后跑 beta=10 的 DiffDPO stage1。
- 在 SC 上用 Slurm 跑同样设置。
- 根据新日志更新 `all_experiments_metric_panels.png`。
- 对比 pair-level `implicit_acc` 和旧 frame-level `implicit_acc`。
- 加 Reg-DPO / winner anchor。

接手者应该先问清楚：

- 是要改训练目标，还是只改诊断？
- 是在 HAL 改代码，还是在 H20/SC 跑训练？
- 新实验要用几张卡、多少 step、哪个数据集、哪个 beta？

## 14. 当前状态快照

截至 2026-05-09：

- HAL 本地代码已经把 DiffDPO stage1/stage2 的 `implicit_acc` 诊断改为 pair-level。
- DPO loss 本身仍是 frame-level。
- 已经运行过语法检查，通过。
- 需要 push 后，H20/SC 才能 pull 到这个逻辑。
- 现有 PRD 里有些历史文件路径过旧，接手时以本文件为最新准。
