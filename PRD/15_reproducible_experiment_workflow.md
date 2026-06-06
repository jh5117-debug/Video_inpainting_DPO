# 可复现实验联动规则

本文件记录 2026-06-06 后必须执行的实验工作流。目标是避免 HAL、H20、PAI、Codex 之间出现代码分叉，尤其避免在 PAI/H20 终端临时改脚本但没有进入 git 的情况。

## 强制顺序

1. HAL 是代码源头。
   - 所有实验脚本、训练逻辑、数据准备工具、评估工具、PRD 和 experiment_registry 更新，必须先在 HAL 的 git worktree 中修改。
   - 不允许把实验关键逻辑只写在 PAI/H20 终端里。

2. 每个实验必须有 registry 文件夹。
   - 文件夹放在 `experiment_registry/<experiment_id_or_name>/`。
   - 至少包含 `README.md`、`config.yaml`、`paths.yaml`、`commands.md`、`status.md`。
   - 大文件、checkpoint、视频、数据集不进 git，只在 registry 中记录路径。

3. HAL 修改后必须 git 化。
   - 先 `git status --short`。
   - 只提交本次实验相关代码和文档。
   - 提交后 `git push origin main`。
   - 如果必须在 PAI/H20 emergency patch，补丁必须立刻回写 HAL、提交并 push。

4. H20 只从 git 接收代码。
   - Codex 可 SSH 到 H20。
   - H20 必须 `git fetch` + `git pull --ff-only` 或在干净 worktree 中 checkout/pull。
   - 如果 H20 有未提交本地文件，先移动到备份或新 worktree；不要 `git reset --hard`。

5. PAI 必须使用同一份已 push 代码。
   - 首选 PAI 直接 `git pull --ff-only origin main`。
   - 如果 PAI 访问 GitHub 不稳定，先让 H20 pull 最新代码，再从 H20 rsync 到 PAI。
   - PAI 上只运行 git 中存在的脚本；不再把大段实验代码粘贴进终端临时执行。

## 服务器职责

- HAL/Codex：读代码、改代码、写 PRD、更新 registry、提交和 push。
- H20：从 git pull 后跑 H20 训练或作为 PAI 同步源。
- PAI：从 git/H20 同步后跑 PAI 训练和验证。

## Exp8 当前约束

Exp8a：

- 已完成 PAI full-loss baseline。
- 结论为负面：Stage1/Stage2 都低于 DiffuEraser-base，不能报告为成功，也不能混称 region-loss。
- 后续不重跑 Exp8a，除非明确只为复现实验产物。

Exp8c：

- 目标：D3 comp loser/mask 不变，只把 winner 换成原始 YouTube-VOS GT，按 `canonical_frame_indices` 对齐。
- loss 不变：`-logsigmoid(-0.5 * 10 * (win_gap - 0.25 * lose_gap)) + 0.05 * m_w + ReLU(win_gap)`。
- H20 因 SIGFPE 使用 `MIXED_PRECISION=no`、全 fp32、`SPLIT_POS_NEG_FORWARD=0`、GPU `1,2,3,4,5,6,7`。
- PAI 不需要 H20 SIGFPE workaround，默认使用 bf16 mixed precision、`SPLIT_POS_NEG_FORWARD=1`、GPU `0,1,2,3,4,5,6,7`。
- PAI 版本必须使用 git 中的：
  - `tools/prepare_exp8c_gtwin_manifest.py`
  - `scripts/launch_exp8c_youtubevos_gtwin_d3comp_fullloss_s1s2_2000_davis_pai.sh`

## 禁止事项

- 禁止在 PAI/H20 只靠终端 sed/python heredoc 修改实验逻辑后直接训练。
- 禁止出现“PAI 修了，HAL/H20/git 没修”的状态。
- 禁止没有 experiment_registry 文件夹就启动新实验。
- 禁止用 VBench 替代 DAVIS inpainting 指标。
- 禁止把 H20 `/home/nvme01/...` 路径放进 PAI 训练 manifest。

## 2026-06-06 Exp7 H20 重启规则

Exp7-fix 重新在 H20 上运行时必须使用 HAL/git 中的可复现脚本：

- `scripts/launch_exp07_fix_smallmask_prior_wingap_s1s2_2000_h20.sh`

当前定义：

- data: `exp07_fix_videodpo_smallmask15_20_prior_k4`
- manifest: H20 本地 `selected_primary_comp.jsonl` 或修复版 `selected_primary_comp.repaired.jsonl`
- task: partial-mask inpainting
- mask: manifest mask, small-mask 15%-20%
- prior/data: ProPainter-prior generated loser
- loss:
  `-logsigmoid(-0.5 * 10 * (win_gap - 0.25 * lose_gap)) + 0.05 * m_w + ReLU(win_gap)`
- stages: Stage1 2000 -> DAVIS `DPO-S1_SFT-S2` validation -> Stage2 2000 -> DAVIS `DPO-S1_DPO-S2` validation
- H20 GPU policy: use `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7`; do not use GPU 0.
- H20 SIGFPE profile: `MIXED_PRECISION=no`, `POLICY_DTYPE=fp32`, `VAE_DTYPE=fp32`, `REF_DTYPE=fp32`, `TEXT_DTYPE=fp32`, `SPLIT_POS_NEG_FORWARD=0`.

H20 main worktree is dirty and must not be reset. Use the clean H20 worktree checked out from pushed HAL/git code, while pointing `OUTPUT_ROOT`, `DATA_ROOT`, and `WEIGHTS_DIR` to `/home/nvme01/H20_Video_inpainting_DPO`.

2026-06-06 14:25 CST status: H20 Exp7-fix Stage1+Stage2 launched from clean
worktree commit `898f9c8` with run version `20260606_142555`. It was monitored
through Stage1 `global_step=20`: `dpo_diagnostics.csv` is present, GPUs 1-7 are
active, GPU 0 is idle, and no SIGFPE/OOM/Traceback was observed.

2026-06-07 02:39 CST correction: the launched H20 Exp7 script completed Stage1
and entered Stage2 but did not perform the intended Stage1 DAVIS validation
before Stage2. This is a workflow miss, not an experiment design change. Recover
with `scripts/run_exp07_fix_smallmask_prior_posthoc_davis_val_h20.sh`, launched
from synced HAL/git code. The watcher waits for Stage2 `last_weights`, uses GPU
1 by default, builds the Stage1 DPO + SFT Stage2 hybrid, then runs both DAVIS
validations:

- Stage1 validation: `DPO-S1_SFT-S2`
- Stage2 validation: `DPO-S1_DPO-S2`

For future Exp7 reruns, the one-shot launcher must either call this watcher or
inline the same validation sequence; do not treat two-stage training alone as a
complete experiment.
