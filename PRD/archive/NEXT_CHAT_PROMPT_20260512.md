# Next Chat Prompt: 2026-05-12

把下面这段直接复制到新的聊天框。

```text
项目路径是：
/home/hj/Video_inpainting_DPO

请你先完整阅读下面这些文件，并把它们当成当前项目最高优先级上下文：

1. /home/hj/Video_inpainting_DPO/PRD/README_FOR_NEXT_CHAT.md
2. /home/hj/Video_inpainting_DPO/PRD/CURRENT_STATUS_20260512.md
3. /home/hj/Video_inpainting_DPO/PRD/NEXT_CHAT_FULL_CONTEXT_20260509.md
4. /home/hj/Video_inpainting_DPO/PRD/PROJECT_HANDOFF_20260509.md
5. /home/hj/Video_inpainting_DPO/PRD/meeting_followup_videodpo_repro_and_bridge_20260511.md
6. /home/hj/Video_inpainting_DPO/PRD/DPO_Training_Metrics_Explained.md
7. /home/hj/Video_inpainting_DPO/PRD/dpo_metric_regularization_prd_20260505.md
8. /home/hj/Video_inpainting_DPO/PRD/code_structure_review.md
9. /home/hj/Video_inpainting_DPO/PRD/DPO_Project_Complete_Summary.md
10. /home/hj/Video_inpainting_DPO/PRD/Project_Complete_Report.md

读完之后，请你先只做一件事：总结你理解到的当前项目状态、实验结论、代码逻辑、HAL/H20/SC 三台服务器工作流、当前训练命令、W&B/VBench/diagnostics 逻辑，以及当前最重要的风险点。不要修改任何文件，等我确认你理解正确后再继续。

特别注意：

- HAL 是开发源头：/home/hj/Video_inpainting_DPO。先在 HAL 改代码、写 PRD、做 smoke，再 commit/push。
- H20 是 H20 GPU 训练机，pull 后用 bash launcher 训练，不用 Slurm。
- SC 是合作者 Slurm 训练机，pull 后用 sbatch 训练。必须保留 PROJECT_HOME/PROJECT_DEV/PROJECT_DATA/WEIGHTS_DIR/WANDB_* 等环境变量路径逻辑，不要硬编码 HAL 或 H20 路径。
- 当前导师本周任务有两条：Task 1 是完整复现 VideoDPO/VC2 并用 VBench 对齐论文指标；Task 2 是从 VideoDPO 出发，先只把模型换成 DiffuEraser full-mask，数据集和 task 暂时保持 VideoDPO。
- 原始 DiffuEraser DPO stage1/stage2 训练代码不要乱动。当前新增的 SC VideoDPO/VC2 代码主要在 DPO_finetune/scripts、tools、patches/videodpo、external submodules 相关路径。
- SC 当前使用 repo-local submodules：external/VideoDPO 和 external/VBench，不再用 sibling naked clone。
- SC health check 已通过；如果要先修环境并检查，运行：
  source ~/.bashrc
  cd "$PROJECT_DEV/Video_inpainting_DPO"
  git pull --ff-only origin main
  CONDA_ENV=diffueraser bash DPO_finetune/scripts/sc_videodpo_fix_env_and_health_check.sh
- SC VideoDPO/VC2 训练脚本是 DPO_finetune/scripts/sc_videodpo_vc2_train.sbatch，Slurm 默认 8 卡 pgpu，W&B 上传到 jh5117-columbia-university/DPO_Diffueraser。
- 最新 SC 训练命令是：
  CONDA_ENV=diffueraser \
  RUN_NAME=sc-vc2-dpo-official-beta5000 \
  BETA_DPO=5000 \
  DPO_DIAG_EVERY=300 \
  sbatch --export=ALL DPO_finetune/scripts/sc_videodpo_vc2_train.sbatch
- 最新提交 f8c68d5 添加了 SC VideoDPO DPO diagnostics。训练脚本默认 APPLY_DPO_DIAG_PATCH=1，会在启动前把 patches/videodpo/sc_vc2_dpo_diagnostics.patch 应用到 external/VideoDPO/lvdm/models/ddpm3d.py。
- 这个 diagnostics patch 保留官方 VideoDPO 的 dpo_loss objective，不改变训练目标；它只把 diagnostics 里的 implicit_acc、mse_w/ref_mse_w、mse_l/ref_mse_l、win_gap、lose_gap、reward_margin、sigma_term、kl_divergence、loser_dominant_ratio 等按 video pair 统计，并每 300 个 optimizer global_step 打印 [dpo_diag] 和更新 W&B 累计表 dpo/diagnostics_table。
- 用户说的 val 指 VBench 论文口径评估，不是 Lightning validation loop。训练后用 DPO_finetune/scripts/sc_videodpo_vc2_checkpoint_sweep.sbatch 对 checkpoint 做 VBench sweep，选择 best 和 last。
- 当前 HAL 工作区可能有用户已有删除状态，不要随手 revert：PRD/First_Finetuning_Summary.md、PRD/PPT.pptx、PRD/stage2_motion_module_init.md、PRD/update_ppt_tables.py、PRD/validation_optimization.md。
- 改任何训练 objective 前必须先确认并写入 PRD；只改 diagnostics 也要写清楚“不改变 objective”。
```

