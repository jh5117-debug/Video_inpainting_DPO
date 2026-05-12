# 新聊天框接手入口

Date: 2026-05-09

如果换一个聊天框继续这个项目，请让新的 Codex 先读这个目录下的文档。推荐顺序如下：

1. `PRD/CURRENT_STATUS_20260512.md`
   - 这是 2026-05-12 的最新状态快照。
   - 包含 SC VideoDPO/VC2 health check 已通过、metadata 误报修复、8 卡 Slurm 训练配置、W&B 对齐、VBench sweep 口径和当前风险点。

2. `PRD/NEXT_CHAT_FULL_CONTEXT_20260509.md`
   - 这是最完整的新聊天框交接入口。
   - 包含三台服务器协作逻辑、当前代码口径、H20/SC 命令、风险点、下一步任务。

3. `PRD/meeting_followup_videodpo_repro_and_bridge_20260511.md`
   - 2026-05-11 导师会议后的最新方向。
   - 包含 Task 1 的 VideoDPO/VC2 VBench 复现脚本，以及 Task 2 的 VideoDPO 数据 + full-mask DiffuEraser 桥接实验。

4. `PRD/PROJECT_HANDOFF_20260509.md`
   - 当前项目状态、三台服务器协作方式、实验结论、代码逻辑、训练命令、风险点。
   - 与 `NEXT_CHAT_FULL_CONTEXT_20260509.md` 互相补充。

5. `PRD/dpo_metric_regularization_prd_20260505.md`
   - 四个实验日志的指标对比。
   - 配套图在 `PRD/assets/dpo_metric_analysis_20260505/all_experiments_metric_panels.png`。

6. `PRD/DPO_Training_Metrics_Explained.md`
   - DPO loss、implicit_acc、win_gap、lose_gap、loser_dominant_ratio 等指标解释。
   - 注意 2026-05-09 后，DiffDPO 的 `implicit_acc` 诊断已经改成 video-pair 粒度。

7. `PRD/DPO_Project_Complete_Summary.md`
   - 早期历史设计总结。
   - 有些路径和代码结构是历史版本，新接手时以 `PROJECT_HANDOFF_20260509.md` 为准。

给新聊天框的推荐开场提示：

```text
请先完整阅读 /home/hj/Video_inpainting_DPO/PRD/README_FOR_NEXT_CHAT.md、/home/hj/Video_inpainting_DPO/PRD/CURRENT_STATUS_20260512.md、/home/hj/Video_inpainting_DPO/PRD/NEXT_CHAT_FULL_CONTEXT_20260509.md 和 /home/hj/Video_inpainting_DPO/PRD/PROJECT_HANDOFF_20260509.md，然后再根据其中的阅读顺序阅读 PRD 里的关键文档和代码。不要直接重构、删除或 revert 文件。先理解当前项目：HAL 本地开发并 push，H20 pull 后用 bash 训练，SC pull 后用 Slurm 训练；当前 DiffDPO 的 implicit_acc 诊断已改成 video-pair 粒度，DPO loss 本身仍保持 frame-level；SC VideoDPO/VC2 health check 已通过，训练脚本当前默认 8 卡、W&B 上传到 jh5117-columbia-university/DPO_Diffueraser。接下来所有修改必须保护现有训练脚本、环境变量路径逻辑和实验日志。
```

重要原则：

- 不要随便改训练服务器路径。
- 不要删除已有日志、checkpoint、PRD assets。
- 不要把 SC 的环境变量路径改成硬编码。
- 不要把 H20 的 bash launcher 改成 Slurm。
- 修改训练逻辑前，先确认改的是诊断指标还是优化目标。
- 改完代码先在 HAL 做 `py_compile` 或 smoke test，再 push 到 Git，训练服务器只 pull。
- SC VideoDPO/VC2 如果要纯官方 `max_epochs=10` 口径，不要传 `MAX_OPT_STEPS`；传了就是内部固定 optimizer-step 对比实验。
