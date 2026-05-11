# Next Chat Full Context: Video Inpainting DPO

Date: 2026-05-09

本文件是给新的聊天框、新的 Codex 终端、或新的合作者接手时使用的完整上下文。新的接手者必须先读本文件，再改代码或跑训练。目标是避免因为缺上下文而误删文件、改错服务器路径、混淆 H20/SC 启动方式、或把诊断指标和训练目标混在一起。

2026-05-11 会议后新增方向见：

```text
PRD/meeting_followup_videodpo_repro_and_bridge_20260511.md
```

该文件是目前最新的任务入口：Task 1 是 VideoDPO/VC2 VBench 复现，Task 2 是在 VideoDPO 数据和任务不变的情况下，用 full-mask 条件把 DiffuEraser 接到这个赛道上。

## 0. 给新聊天框的开场提示

如果要换一个聊天框，建议直接粘贴下面这段：

```text
请先完整阅读 /home/hj/Video_inpainting_DPO/PRD/README_FOR_NEXT_CHAT.md、/home/hj/Video_inpainting_DPO/PRD/NEXT_CHAT_FULL_CONTEXT_20260509.md、/home/hj/Video_inpainting_DPO/PRD/PROJECT_HANDOFF_20260509.md，然后按 README 里的顺序阅读 PRD 关键文档。不要直接重构、删除、revert 或硬编码路径。

当前项目的开发逻辑是：
1. HAL 服务器 /home/hj/Video_inpainting_DPO 是主要开发机，上面有 Codex，用来读代码、改代码、写 PRD、分析日志；改完后 push 到 Git。
2. H20 服务器有 H20 GPU，只负责 pull 最新代码后用 bash 脚本训练；不要把 H20 改成 Slurm。
3. SC 服务器是合作者训练资源，必须 pull 最新代码后用 Slurm/sbatch；不要把 SC 的环境变量路径改成硬编码。
4. 当前 DiffDPO stage1/stage2 的 implicit_acc 诊断已经改成 video-pair 粒度；DPO loss 本身仍保持 frame-level 训练目标。修改训练逻辑前必须先说明是改 diagnostics 还是改 objective。

先读文档，再运行 git status --short。不要删除日志、checkpoint、PRD assets，也不要 revert 用户已有改动。
```

## 1. 项目一句话

项目目标是在 Video Inpainting / DiffuEraser 的 SFT 基础上引入 DPO / VideoDPO 风格偏好优化，比较 DiffuEraser 基底和 VideoDPO 基底在 video inpainting preference pair 上的训练行为，并解决普通 vanilla DPO 容易出现的 loser-dominant failure。

当前核心判断：

- 普通 DiffDPO 很容易出现 `implicit_acc≈1`、`DPO_loss≈0`，但 PSNR/SSIM 下降。
- 这不是训练成功，而是模型可能主要通过让 loser 更差来满足相对排序。
- `implicit_acc` 不能单独作为成功信号，必须结合 `win_gap`、`lose_gap`、`mse_w/ref_mse_w`、`mse_l/ref_mse_l`、`loser_dominant_ratio`、PSNR/SSIM。
- 2026-05-09 后，DiffDPO 的 `implicit_acc` 诊断口径已经从 frame-level 改成 video-pair-level。

## 2. 三台服务器的职责

### 2.1 HAL：开发源头

当前本地路径：

```text
/home/hj/Video_inpainting_DPO
```

HAL 的职责：

- 用 Codex 开发代码。
- 写 PRD 和交接文档。
- 解析日志、生成图表。
- 运行语法检查或小 smoke test。
- commit/push 到 GitHub。

HAL 的原则：

- HAL 是主要开发源头。
- 不要在 H20/SC 上临时乱改训练代码再忘记同步。
- 修改后先检查：

```bash
cd /home/hj/Video_inpainting_DPO
git status --short
python -m py_compile training/dpo/train_stage1.py training/dpo/train_stage2.py training/dpo/scripts/run_stage1.py
```

当前需要特别注意：工作区可能有已有未提交改动和 untracked PRD assets。不要随便 `git reset --hard`，不要删除不认识的文件。

### 2.2 H20：GPU 训练机，bash 启动

常见路径：

```text
/home/nvme01/Video_inpainting_DPO              # DiffDPO / DiffuEraser 项目
/home/nvme01/H20_Video_inpainting_DPO          # VideoDPO + VideoInpainting adapter 项目
/home/nvme01/VideoDPO                          # 开源 VideoDPO repo，被 adapter patch
```

H20 的职责：

- `git pull` 最新代码。
- 直接用 bash 脚本跑训练。
- 用户通常自己在 H20 终端执行命令。

H20 不走 Slurm。不要给 H20 写 `sbatch` 命令。

DiffDPO stage1 beta=10 推荐命令：

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

后台版本：

```bash
cd /home/nvme01/Video_inpainting_DPO
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_GPUS=8 BATCH_SIZE=1 \
BETA_DPO=10 DPO_LOSE_GAP_WEIGHT=1.0 MAX_STEPS=10000 CKPT_STEPS=2000 VAL_STEPS=2000 \
RUN_NAME=h20-dpo-stage1-beta10-pair-implicit-acc MIXED_PRECISION=bf16 \
SPLIT_POS_NEG_FORWARD=1 GRADIENT_CHECKPOINTING=1 \
bash scripts/h20_run_dpo_stage1.sh > logs/h20_dpo_stage1_beta10_pairacc.out 2>&1 &
```

查看：

```bash
tail -f logs/h20_dpo_stage1_beta10_pairacc.out
```

VideoDPO + Inpainting adapter 使用：

```bash
cd /home/nvme01/H20_Video_inpainting_DPO
git pull --ff-only origin main
bash patches/videodpo/apply_and_launch_h20_videodpo.sh
```

如果要用 H20 上 GPU `2,3,4,5,6,7`，确认脚本是否已经默认 `CUDA_VISIBLE_DEVICES=2,3,4,5,6,7`，不要假设。

### 2.3 SC：合作者 Slurm 训练机

SC 是合作者训练资源，要求严格。SC 上不能硬编码个人路径，必须尊重现有环境变量设计。

原始 SC 用法：

```bash
source ~/.bashrc
cd "$PROJECT_DEV/Video_inpainting_DPO"
git pull --ff-only origin main
mkdir -p logs
sbatch --export=ALL DPO_finetune/scripts/03_dpo_stage1.sbatch
```

推荐封装：

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

或：

```bash
BETA_DPO=10 \
MAX_STEPS=10000 \
RUN_NAME=sc-dpo-stage1-beta10-pair-implicit-acc \
bash scripts/sc_submit_dpo_stage1.sh
```

SC 路径环境变量：

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

必须保护这些环境变量逻辑。不要把 H20 的 `/home/nvme01/...` 写进 SC 脚本。

## 3. 当前代码逻辑：DiffDPO implicit_acc 已改成 pair-level diagnostics

核心文件：

```text
training/dpo/train_stage1.py
training/dpo/train_stage2.py
```

2026-05-09 最新改动：

- `compute_dpo_loss(...)` 新增 `nframes` 参数。
- DPO loss 本身保持原来的 frame-level 优化目标。
- diagnostics 里的 `implicit_acc`、`inside_term_mean/min/max`、`Samples correct/total`、`loser_dominant_ratio` 改为 video-pair 粒度。
- stage1 和 stage2 都传入 `nframes=args.nframes`。

当前逻辑可以理解为：

```text
per-frame win_gap / lose_gap
    ↓
frame_inside_term 用于 DPO loss
    ↓
按 nframes reshape，平均成 pair_win_gap / pair_lose_gap
    ↓
pair-level inside_term 用于 implicit_acc diagnostics
    ↓
跨卡 gather 后得到全局 Samples: correct/total
```

重要：现在只改了 diagnostics，没有改训练 objective。

如果未来要把 DPO loss 也改成 pair-level，必须作为新实验写清楚，因为它会改变优化目标，不能和旧 DiffDPO loss 直接混比。

### 3.1 判断正确的 pair 是什么

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

这个 pair 就算判断正确。

普通 DPO 下直观上接近：

```text
win_gap < lose_gap
```

这说明 policy 相对 ref 更偏向 winner。但它不保证 winner 真变好，因为也可能是 loser 变得更差。

### 3.2 为什么 pair-level 更合理

偏好标签本质是：

```text
这个 winner 视频 比这个 loser 视频好
```

而不是每一帧独立给标签。

旧 frame-level 统计会把 8 卡、batch_size=1、16 帧变成：

```text
8 * 1 * 16 = 128 个 frame 判断
```

新 pair-level 统计是：

```text
8 * 1 = 8 个 video pair 判断
```

因此新日志应看到类似：

```text
Samples: 4/8 correct
implicit_acc = 0.5
```

而不是旧的：

```text
Samples: 64/128 correct
```

## 4. 当前指标和四个实验的解释

主报告：

```text
PRD/dpo_metric_regularization_prd_20260505.md
```

主图：

```text
PRD/assets/dpo_metric_analysis_20260505/all_experiments_metric_panels.png
```

四个日志：

```text
/home/hj/log/普通DiffDPO_loss.log
/home/hj/log/把lose_gap删除的loss.log
/home/hj/log/VideoDPO的训练.log
/home/hj/log/使用VideoInpainting的数据集的VideoDPO的loss.log
```

四个实验：

| 实验 | 基底 | 数据/目标 | 当前判断 |
|---|---|---|---|
| DiffDPO_loss | DiffuEraser | 普通 DPO | loser-dominant failure，DPO_loss 贴 0，PSNR/SSIM 下降 |
| no-lose-gap ablation | DiffuEraser | 删除 loser gap / winner-only | PSNR/SSIM 稳定，但不能等价于标准 DPO |
| VideoDPO open-source | VideoDPO | 原始 VideoDPO 数据 | 曲线更平滑，gap 量级小 |
| VideoDPO + VideoInpainting data | VideoDPO | inpainting winner/loser pair | 也出现 loser-dominant 倾向，缺少有效 PSNR/SSIM |

不要把 `implicit_acc=1` 当成功。真正危险组合是：

```text
implicit_acc≈1
DPO_loss≈0
win_gap > 0
lose_gap >> win_gap
loser_dominant_ratio≈1
PSNR/SSIM 下降
```

这表示 DPO 可能通过恶化 loser 来赢。

## 5. DPO objective 是什么

这里 objective 指训练时最小化的目标函数，也就是 loss。

普通 Diffusion-DPO：

```text
DPO_loss = -logsigmoid(-0.5 * beta * (win_gap - lose_gap))
```

危险点：

- 它只要求相对排序赢。
- 可以通过 winner 变好来赢。
- 也可以通过 loser 变差来赢。
- 当前普通 DiffDPO 的失败就属于后者。

因此后续推荐加入：

```text
L_total =
  L_DPO_norm
  + lambda_a * m_w
  + lambda_w * ReLU(m_w - m_w_ref)
  + lambda_g * ReLU(tilde_lose_gap - tilde_win_gap - tau_g)
```

也就是 winner anchor / DPOP / Reg-DPO / anti-loser-dominance 一类约束。

## 6. 当前重要代码文件

训练：

```text
training/dpo/train_stage1.py
training/dpo/train_stage2.py
training/dpo/scripts/run_stage1.py
training/dpo/scripts/run_stage2.py
training/dpo/dataset/dpo_dataset.py
```

H20：

```text
scripts/h20_run_dpo_stage1.sh
scripts/h20_run_dpo_stage2.sh
scripts/h20_smoke_dpo_stage1.sh
patches/videodpo/apply_and_launch_h20_videodpo.sh
patches/videodpo/h20_videoinpaint_dpo_adapter.patch
```

SC：

```text
DPO_finetune/scripts/03_dpo_stage1.sbatch
DPO_finetune/scripts/03_dpo_stage2.sbatch
scripts/sc_submit_dpo_stage1.sh
```

PRD：

```text
PRD/README_FOR_NEXT_CHAT.md
PRD/NEXT_CHAT_FULL_CONTEXT_20260509.md
PRD/PROJECT_HANDOFF_20260509.md
PRD/DPO_Training_Metrics_Explained.md
PRD/dpo_metric_regularization_prd_20260505.md
PRD/code_structure_review.md
PRD/DPO_Project_Complete_Summary.md
PRD/Project_Complete_Report.md
```

分析脚本：

```text
PRD/scripts/analyze_dpo_metric_logs.py
```

注意：如果重新运行 `PRD/scripts/analyze_dpo_metric_logs.py`，它会重写 `PRD/dpo_metric_regularization_prd_20260505.md`。脚本必须保留 2026-05-09 的 pair-level 口径说明，避免覆盖掉最新交接信息。

## 7. 旧文档和新文档的关系

有些旧 PRD 是历史版本，路径或代码结构可能过时：

- `DPO_Project_Complete_Summary.md`
- `Project_Complete_Report.md`
- `code_structure_review.md`
- `sft_pipeline_enhancement.md`

这些文档仍有价值，但接手时以这几个为最新准：

1. `PRD/README_FOR_NEXT_CHAT.md`
2. `PRD/NEXT_CHAT_FULL_CONTEXT_20260509.md`
3. `PRD/PROJECT_HANDOFF_20260509.md`
4. `PRD/DPO_Training_Metrics_Explained.md`
5. `PRD/dpo_metric_regularization_prd_20260505.md`

## 8. 安全规则

必须遵守：

1. 先读 PRD，再改代码。
2. 先 `git status --short`，确认已有改动。
3. 不要 revert 用户已有改动。
4. 不要删除日志、checkpoint、PRD assets。
5. 不要硬编码 SC 路径。
6. 不要把 H20 bash launcher 改成 Slurm。
7. 不要把 SC Slurm 脚本改成 H20 bash 逻辑。
8. 修改训练代码后跑 `py_compile`。
9. 新实验必须记录 beta、step、GPU 数、batch_size、数据路径、代码 commit。
10. 修改 loss 时必须写入 PRD，因为这是新实验定义。
11. 修改 diagnostics 时必须写清楚不改变训练目标。
12. 不要只凭 `implicit_acc` 或 `DPO_loss` 判断实验成功。

## 9. 新聊天框应先执行的阅读命令

在 HAL 上：

```bash
cd /home/hj/Video_inpainting_DPO
find PRD -maxdepth 3 -type f | sort
sed -n '1,220p' PRD/README_FOR_NEXT_CHAT.md
sed -n '1,260p' PRD/NEXT_CHAT_FULL_CONTEXT_20260509.md
sed -n '1,320p' PRD/PROJECT_HANDOFF_20260509.md
sed -n '1,240p' PRD/DPO_Training_Metrics_Explained.md
sed -n '1,220p' PRD/dpo_metric_regularization_prd_20260505.md
git status --short
```

如果要继续读完整文档，再按 README 的顺序读剩余段落。

## 10. 下一步最可能任务

短期：

- push HAL 上 pair-level diagnostics 改动。
- H20 pull 后跑 DiffDPO stage1 beta=10。
- SC pull 后用 Slurm 跑同样配置。
- 观察新 pair-level `implicit_acc` 是否不再虚高。
- 更新 `all_experiments_metric_panels.png` 和 CSV。

中期：

- 加 winner-side SFT / Reg-DPO anchor。
- 加 DPOP-style `ReLU(m_w - m_w_ref)`。
- 加 anti-loser-dominance 正则。
- 修 VideoDPO + Inpainting validation 的 `skimage` / PSNR / SSIM 输出。

长期：

- 改造更合理的 hard-but-plausible loser 数据，而不是只用 absolute worst loser。
- 更 mask-aware / region-aware 的 DPO loss。
- 对比 DiffuEraser 与更强基底如 VideoDPO / DiT / Flow Matching。
