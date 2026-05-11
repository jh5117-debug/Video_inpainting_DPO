# Video Inpainting DPO Diagnostics and Regularized-DPO PRD

Date: 2026-05-07

2026-05-09 update:

- 最新项目交接入口见 `PRD/README_FOR_NEXT_CHAT.md`、`PRD/NEXT_CHAT_FULL_CONTEXT_20260509.md` 和 `PRD/PROJECT_HANDOFF_20260509.md`。
- HAL 本地代码已经把 DiffDPO stage1/stage2 的 `implicit_acc` 诊断改成 video-pair 粒度。
- 当前改动只影响 diagnostics，不改变 DPO loss 本身的 frame-level 优化目标。
- 本文中的历史实验图仍来自旧日志，其中 DiffDPO 旧日志的 `implicit_acc` 是 frame-level 统计；新实验需要用更新后的 pair-level 口径重新画图。

Source logs:

- `/home/hj/log/普通DiffDPO_loss.log`
- `/home/hj/log/把lose_gap删除的loss.log`
- `/home/hj/log/VideoDPO的训练.log`
- `/home/hj/log/使用VideoInpainting的数据集的VideoDPO的loss.log`

Generated artifacts:

- `assets/dpo_metric_analysis_20260505/all_experiments_metric_panels.png`
- `assets/dpo_metric_analysis_20260505/comparison_overlay_smoothed.png`
- `assets/dpo_metric_analysis_20260505/videodpo_300step_metric_panels.png`
- `assets/dpo_metric_analysis_20260505/videodpo_inpainting_data_300step_metric_panels.png`
- CSV tables in `assets/dpo_metric_analysis_20260505/`

![All metric panels](assets/dpo_metric_analysis_20260505/all_experiments_metric_panels.png)

![Smoothed comparison](assets/dpo_metric_analysis_20260505/comparison_overlay_smoothed.png)

![VideoDPO 300-step panels](assets/dpo_metric_analysis_20260505/videodpo_300step_metric_panels.png)

![VideoDPO on VideoInpainting data 300-step panels](assets/dpo_metric_analysis_20260505/videodpo_inpainting_data_300step_metric_panels.png)

## 1. Executive Summary

这四个真实训练日志现在给出的结论更清楚了：

1. **普通 DiffDPO_loss 出现了典型的 loser-dominant failure。** 训练目标很快被满足，`implicit_acc` 进入接近 1 的区域，`dpo_loss` 贴近 0；但 `win_gap` 和 `lose_gap` 同时变大，且 `lose_gap` 远大于 `win_gap`。这说明模型主要靠把 loser 变得更差来赢。在当前 `global-step<=10000` 对齐窗口内，PSNR/SSIM 从最早 validation 的 `20.4550 / 0.7779` 掉到最后的 `14.7542 / 0.5737`。

2. **删除 lose_gap 的 ablation 证明 winner 分支本身不是坏的。** 最后 20% 诊断里 `win_gap=-0.000600`，仍然小于 0；PSNR/SSIM 保持在 `24.19 / 0.866` 左右。这说明当不允许模型靠 loser 侧获利时，winner 侧可以稳定工作。

3. **开源 VideoDPO 也不是“winner MSE 必须优于 ref”的形态。** 它最后 20% 的 `win_gap=0.002634`、`lose_gap=0.003421`，二者都为正，但尺度只有 1e-3 量级。VideoDPO 的偏好优化成功不是因为每一步 denoising MSE 都比 ref 好，而是最终 sampled video 的偏好指标更好。对我们这个 video inpainting 任务，不能直接把 VideoDPO 的现象当作安全证明，因为我们的普通 DiffDPO 已经在 PSNR/SSIM 上实际崩了。

4. **VideoDPO 换成 VideoInpainting 数据后，也按 global-step 直接看，不再按 epoch 裁剪。** 这里的横坐标统一使用训练框架里的 optimizer `global_step`。最新日志里既有每 300 个 global-step 的 `[dpo_diag]`，也有 Lightning 进度条里的逐步 `rank0/dpo_loss / win_gap / lose_gap`；本报告把二者合并到同一个 global-step 轴上。当前已完成日志最后 20% 的 `win_gap=0.012829`、`lose_gap=0.023730`，`loser_dominant_ratio=1.000000`，说明统一数据集后 VideoDPO 仍然会出现强烈的 loser shortcut 倾向。

因此下一步不应该继续裸跑普通 DPO，而应该加入 **winner anchor + DPOP/Reg-DPO/APO 风格正则化**，防止目标函数只通过 loser 侧退化获得高 `implicit_acc`。第四个实验还额外说明：问题不只是 DiffuEraser 的实现问题，也和 VideoInpainting pair 的局部差异、mask 区域质量、winner/loser 可分性有关。

## 2. Experiment Contents and Per-Experiment Summaries

### 2.1 普通 DiffDPO_loss

**Log:** `/home/hj/log/普通DiffDPO_loss.log`

**实验内容：** 这是 DiffuEraser / Video Inpainting DPO 的普通 baseline。训练目标保留标准 DPO 排序项，winner 和 loser 两侧都参与偏好差值计算；当前分析窗口使用 `global-step<=10000`，诊断来自 `DPO Diagnostics @ Step ...`，并且有 validation PSNR/SSIM。

**关键结果：**

- `dpo_loss` 从 `0.694824` 降到 `0.000000`，DPO 排序目标很快被优化到接近 0。
- `implicit_acc` 从 `0.429688` 升到 `1.000000`，最后 20% 均值为 `1.000000`。
- `win_gap` 从 `0.000015` 增大到 `0.753472`；`lose_gap` 从 `0.000002` 增大到 `0.985496`。
- 最后 20% 的 `lose_gap - win_gap=0.214243`，`loser_dominant_ratio=1.000000`。
- PSNR/SSIM 从 `20.4550 / 0.7779` 下降到 `14.7542 / 0.5737`。

**实验总结：** 这个实验证明裸 DPO 可以非常快地满足相对偏好排序，但满足方式是危险的：policy 并没有稳定改善 winner，反而同时拉高 winner/loser 的误差，并且更强地拉高 loser 误差。`implicit_acc=1` 和 `dpo_loss≈0` 在这里不是质量提升信号，而是 loser shortcut 的表征；PSNR/SSIM 的下降说明这个 shortcut 已经转化为真实 inpainting 质量退化。

### 2.2 删除 lose_gap 的 DiffDPO ablation

**Log:** `/home/hj/log/把lose_gap删除的loss.log`

**实验内容：** 这是 DiffDPO 的 ablation，训练 loss 中删除 loser gap 相关贡献，只保留 winner 侧约束；但日志仍然继续计算 `ml/mrefl/lose_gap` 作为 monitor-only 指标，用来观察 policy 对 loser 分支的副作用。分析窗口同样使用 `global-step<=10000`，并有 validation PSNR/SSIM。

**关键结果：**

- `dpo_loss` 基本保持在 `0.693042 -> 0.689204`，没有像普通 DiffDPO 那样塌到 0。
- 最后 20% 的 `win_gap=-0.000600`，仍然小于 0，说明 winner 侧没有被推坏。
- 最后 20% 的 `lose_gap=0.001330`，`lose_gap - win_gap=0.001930`，远小于普通 DiffDPO 的 loser-dominant 幅度。
- PSNR/SSIM 从 `24.1625 / 0.8658` 到 `24.1936 / 0.8665`，基本稳定。

**实验总结：** 删除 loser gap 后，winner reconstruction 本身可以稳定工作，质量指标也没有崩。这说明问题不是数据完全不可学，也不是 winner 分支天然坏掉；问题主要来自普通 DPO 允许模型通过扩大 loser 误差来赢。这个实验给后续正则化方向提供了最重要的对照：应该保留偏好学习，但必须给 winner 侧加 anchor，并限制 loser-dominant shortcut。

### 2.3 开源 VideoDPO 原始训练日志

**Log:** `/home/hj/log/VideoDPO的训练.log`

**实验内容：** 这是开源 VideoDPO 代码的训练日志。日志每个 optimizer `global_step` 输出 `[dpo_diag]`，本报告直接使用 `global_step` 作为横坐标，范围为 `0-9999`。该日志没有 Video Inpainting 的 PSNR/SSIM validation，因此只能比较 DPO 中间指标。

**关键结果：**

- `dpo_loss` 从 `0.698669` 降到 `0.178772`，但不像普通 DiffDPO 那样长期贴 0。
- `implicit_acc` 从 `0.546875` 到 `0.937500`；最后 20% 均值为 `0.687582`。
- 最后 20% 的 `win_gap=0.002634`，`lose_gap=0.003421`，二者为正但只有 1e-3 量级。
- `loser_dominant_ratio` 从前 20% 的 `0.736759` 上升到最后 20% 的 `0.992039`。

**实验总结：** 开源 VideoDPO 的 DPO 训练并不要求每个 winner denoising MSE 都优于 reference；后期 `win_gap` 也可以略大于 0。关键区别是它的 gap 漂移量级很小，而且原始 VideoDPO 关注最终 sampled video 的偏好质量，不是单步 epsilon MSE 本身。因此它不能直接证明我们的 inpainting DPO 是安全的。它更像一个参考：VideoDPO 的相对偏好优化会自然产生小幅正 gap，但普通 DiffDPO 的 gap 扩大到了 1e-1 到 1e0，并且伴随 PSNR/SSIM 崩溃，这已经不是同一类现象。

### 2.4 VideoDPO 使用 VideoInpainting 数据

**Log:** `/home/hj/log/使用VideoInpainting的数据集的VideoDPO的loss.log`

**实验内容：** 这是把开源 VideoDPO 训练代码适配到 Video Inpainting winner/loser pair 后得到的日志。winner 主要来自 GT / clean target，loser 来自 inpainting 模型输出。最新日志解析到 `global_step=6101`：其中 `[dpo_diag]` 每 300 个 global-step 输出一次完整指标，Lightning 进度条则提供更密的 `dpo_loss / implicit_acc / win_gap / lose_gap`。PSNR/SSIM validation 曾尝试运行，但旧 H20 环境缺少 `skimage`，因此没有成功写出质量指标。

**关键结果：**

- `dpo_loss` 从起点 `0.705480` 到最新单点 `0.673000`；最后 20% 均值为 `0.169640`，说明偏好目标整体被压低，但尾部仍存在 batch-level spike。
- `implicit_acc` 从 `0.500000` 到 `1.000000`；最后 20% 均值为 `0.959839`。
- `win_gap` 从 `0.000005` 增大到 `0.015900`；`lose_gap` 从 `-0.000005` 增大到 `0.015900`。
- 最后 20% 的 `lose_gap - win_gap=0.010901`，`loser_dominant_ratio=1.000000`。
- validation failure 明确记录为 `ModuleNotFoundError: No module named 'skimage'`，所以这个实验目前仍没有可用 PSNR/SSIM。

**实验总结：** 统一到 Video Inpainting 数据后，开源 VideoDPO 也出现了明显的 loser-dominant 倾向：`lose_gap` 大多数时间高于 `win_gap`，并且 `[dpo_diag]` 给出的 `loser_dominant_ratio` 后期接近 1。这个结果说明风险不只是 DiffuEraser 某段实现的偶然问题，而是 Video Inpainting pair 的局部差异和普通 DPO 相对排序目标之间存在结构性冲突。由于最新日志仍没有 PSNR/SSIM，这个实验不能单独判断最终视觉质量，但它已经足够说明：只靠裸 DPO 排序，不足以保证 inpainting 质量会变好。

## 3. Experiment Scope and Step Setting

主对比使用如下数据窗口：

| experiment | raw_rows | used_rows | raw_step_range | used_step_range | analysis_cutoff_step | analysis_note |
| --- | --- | --- | --- | --- | --- | --- |
| DiffDPO_loss | 44 | 34 | 1-12900 | 1-9900 | 10000 | global-step aligned to step<=10000. |
| DiffDPO_no_lose_gap | 34 | 34 | 1-9900 | 1-9900 | 10000 | global-step aligned to step<=10000. |
| VideoDPO_open_source | 10000 | 10000 | 0-9999 | 0-9999 | 10000 | old log records optimizer global_step; use it directly as the x-axis. |
| VideoDPO_on_VideoInpainting_data | 6102 | 6102 | 0-6101 | 0-6101 | 10000 | old log records optimizer global_step; use it directly as the x-axis. New mixed logs combine [dpo_diag] rows and Lightning progress rows, both aligned to optimizer global_step. |

关键点：

- Lightning 进度条里的 `Epoch 0: 0/1785` 只是 dataloader 的 batch 进度，不作为本报告横坐标。
- 本报告横坐标统一叫 `global-step`，含义是训练框架里的 optimizer `global_step`。
- 对 DiffDPO 两个日志，原始 `DPO Diagnostics @ Step` 按 global-step 使用；对 VideoDPO 日志，优先读取 `[dpo_diag] global_step=...`，并把 Lightning 进度条还原为 `epoch * steps_per_epoch + batch_idx` 后合并到同一 global-step 轴。
- 完整解析结果保存在 `/home/hj/Video_inpainting_DPO/PRD/assets/dpo_metric_analysis_20260505/all_diagnostics_full.csv`；主图和主表使用 `/home/hj/Video_inpainting_DPO/PRD/assets/dpo_metric_analysis_20260505/all_diagnostics.csv`。

## 4. Metric Definitions

当前诊断口径：

- `mw` / `mse_w`: policy 在 winner/GT 样本上的 epsilon MSE。
- `ml` / `mse_l`: policy 在 loser/model-output 样本上的 epsilon MSE。
- `mrefw` / `ref_mse_w`: reference 在 winner 上的 epsilon MSE。
- `mrefl` / `ref_mse_l`: reference 在 loser 上的 epsilon MSE。
- `win_gap = mw - mrefw`。小于 0 表示 policy 在 winner denoising MSE 上优于 ref。
- `lose_gap = ml - mrefl`。大于 0 表示 policy 在 loser denoising MSE 上差于 ref。
- 普通 DPO 的核心判断近似是 `win_gap < lose_gap`，而不是要求 `win_gap < 0`。
- `implicit_acc` 是当前 batch / 当前 timestep / 当前 noise 上 `inside_term > 0` 的比例，分布式时会 gather 所有 rank；它不是累计平均。

重要解释：`implicit_acc` 高只说明相对排序被满足，不说明 winner 质量变好。普通 DiffDPO 的日志已经证明了这一点：`implicit_acc` 可以接近 1，同时 PSNR/SSIM 大幅下降。

## 5. Phase Summary

每个实验取诊断记录的前 20% 和后 20% 做均值：

| experiment | phase | step_range | n | l_dpo | implicit_acc | win_gap | lose_gap | mse_w | ref_mse_w | mse_l | ref_mse_l | loser_dominant_ratio | lose_minus_win_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DiffDPO_loss | first_20pct | 1-1800 | 7 | 0.099871 | 0.912946 | 0.115116 | 0.177192 | 0.206482 | 0.091366 | 0.271082 | 0.093890 | 0.937662 | 0.062076 |
| DiffDPO_loss | last_20pct | 8100-9900 | 7 | 0.000183 | 1.000000 | 0.846902 | 1.061145 | 0.861756 | 0.014854 | 1.074960 | 0.013815 | 1.000000 | 0.214243 |
| DiffDPO_no_lose_gap | first_20pct | 1-1800 | 7 | 0.691437 | 0.994643 | -0.000686 | 0.001953 | 0.029212 | 0.029898 | 0.162622 | 0.160668 | 0.569236 | 0.002639 |
| DiffDPO_no_lose_gap | last_20pct | 8100-9900 | 7 | 0.691649 | 1.000000 | -0.000600 | 0.001330 | 0.009949 | 0.010549 | 0.051766 | 0.050436 | 0.596429 | 0.001930 |
| VideoDPO_open_source | first_20pct | 0-1999 | 2000 | 1.208438 | 0.527652 | 0.000243 | 0.000312 | 0.057136 | 0.056892 | 0.055941 | 0.055629 | 0.736759 | 0.000069 |
| VideoDPO_open_source | last_20pct | 8000-9999 | 2000 | 1.141425 | 0.687582 | 0.002634 | 0.003421 | 0.061086 | 0.058452 | 0.057212 | 0.053791 | 0.992039 | 0.000787 |
| VideoDPO_on_VideoInpainting_data | first_20pct | 0-1220 | 1221 | 0.230776 | 0.904415 | 0.001556 | 0.004270 | 0.061023 | 0.059723 | 0.100891 | 0.097828 | 0.933333 | 0.002714 |
| VideoDPO_on_VideoInpainting_data | last_20pct | 4881-6101 | 1221 | 0.169640 | 0.959839 | 0.012829 | 0.023730 | 0.017976 | 0.002766 | 0.099797 | 0.078220 | 1.000000 | 0.010901 |

## 6. Start-to-End Delta

| experiment | first_step | last_step | l_dpo_first | l_dpo_last | l_dpo_delta | l_dpo_first_step | l_dpo_last_step | implicit_acc_first | implicit_acc_last | implicit_acc_delta | implicit_acc_first_step | implicit_acc_last_step | win_gap_first | win_gap_last | win_gap_delta | win_gap_first_step | win_gap_last_step | lose_gap_first | lose_gap_last | lose_gap_delta | lose_gap_first_step | lose_gap_last_step | mse_w_first | mse_w_last | mse_w_delta | mse_w_first_step | mse_w_last_step | mse_l_first | mse_l_last | mse_l_delta | mse_l_first_step | mse_l_last_step | ref_mse_w_first | ref_mse_w_last | ref_mse_w_delta | ref_mse_w_first_step | ref_mse_w_last_step | ref_mse_l_first | ref_mse_l_last | ref_mse_l_delta | ref_mse_l_first_step | ref_mse_l_last_step | psnr_first | psnr_last | psnr_best | ssim_first | ssim_last | ssim_best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DiffDPO_loss | 1 | 9900 | 0.694824 | 0.000000 | -0.694824 | 1 | 9900 | 0.429688 | 1.000000 | 0.570312 | 1 | 9900 | 0.000015 | 0.753472 | 0.753457 | 1 | 9900 | 0.000002 | 0.985496 | 0.985494 | 1 | 9900 | 0.123636 | 0.777629 | 0.653993 | 1 | 9900 | 0.156430 | 1.007462 | 0.851032 | 1 | 9900 | 0.123622 | 0.024157 | -0.099465 | 1 | 9900 | 0.156428 | 0.021967 | -0.134461 | 1 | 9900 | 20.455000 | 14.754200 | 20.455000 | 0.777900 | 0.573700 | 0.777900 |
| DiffDPO_no_lose_gap | 1 | 9900 | 0.693042 | 0.689204 | -0.003838 | 1 | 9900 | 0.987500 | 1.000000 | 0.012500 | 1 | 9900 | -0.000042 | -0.001581 | -0.001539 | 1 | 9900 | 0.000195 | 0.003557 | 0.003362 | 1 | 9900 | 0.022386 | 0.005808 | -0.016578 | 1 | 9900 | 0.169708 | 0.034778 | -0.134930 | 1 | 9900 | 0.022428 | 0.007389 | -0.015039 | 1 | 9900 | 0.169512 | 0.031221 | -0.138291 | 1 | 9900 | 24.162500 | 24.193600 | 24.194600 | 0.865800 | 0.866500 | 0.866600 |
| VideoDPO_open_source | 0 | 9999 | 0.698669 | 0.178772 | -0.519897 | 0 | 9999 | 0.546875 | 0.937500 | 0.390625 | 0 | 9999 | -0.000010 | 0.001535 | 0.001545 | 0 | 9999 | 0.000012 | 0.002412 | 0.002400 | 0 | 9999 | 0.013123 | 0.008641 | -0.004482 | 0 | 9999 | 0.026521 | 0.019358 | -0.007163 | 0 | 9999 | 0.013134 | 0.007106 | -0.006028 | 0 | 9999 | 0.026510 | 0.016947 | -0.009563 | 0 | 9999 |  |  |  |  |  |  |
| VideoDPO_on_VideoInpainting_data | 0 | 6101 | 0.705480 | 0.673000 | -0.032480 | 0 | 6101 | 0.500000 | 1.000000 | 0.500000 | 0 | 6101 | 0.000005 | 0.015900 | 0.015895 | 0 | 6101 | -0.000005 | 0.015900 | 0.015905 | 0 | 6101 | 0.015590 | 0.023744 | 0.008154 | 0 | 6000 | 0.029767 | 0.028337 | -0.001430 | 0 | 6000 | 0.015586 | 0.004359 | -0.011227 | 0 | 6000 | 0.029772 | 0.009168 | -0.020604 | 0 | 6000 |  |  |  |  |  |  |

## 7. Validation-Aligned Table

DiffuEraser 两个实验有 PSNR/SSIM。开源 VideoDPO 日志没有 PSNR/SSIM。第四个 VideoDPO+Inpainting 实验尝试了 `video_inpaint_val`，但 H20 的 VideoDPO 环境缺少 `skimage`，所以没有成功写出 PSNR/SSIM；这意味着目前不能把第四个实验和前两个实验做质量指标的公平数值对比，只能比较中间 DPO diagnostic。

| experiment | val_step | nearest_diag_step | l_dpo | implicit_acc | win_gap | lose_gap | mse_w | mse_l | ref_mse_w | ref_mse_l | loser_dominant_ratio | psnr | ssim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DiffDPO_loss | 2000 | 2100 | 0.000000 | 0.984375 | 0.342837 | 0.674264 | 0.560823 | 0.863167 | 0.217987 | 0.188903 | 1.000000 | 20.455000 | 0.777900 |
| DiffDPO_loss | 4000 | 3900 | 0.000000 | 1.000000 | 0.146834 | 0.268983 | 0.596270 | 0.723834 | 0.449437 | 0.454851 | 1.000000 | 19.150700 | 0.733400 |
| DiffDPO_loss | 6000 | 6000 | 0.000000 | 1.000000 | 0.411343 | 0.788650 | 0.525594 | 0.900120 | 0.114250 | 0.111470 | 1.000000 | 19.151300 | 0.734900 |
| DiffDPO_loss | 8000 | 8100 | 0.001249 | 1.000000 | 1.015003 | 1.108772 | 1.019947 | 1.113991 | 0.004944 | 0.005220 | 1.000000 | 16.578000 | 0.642800 |
| DiffDPO_loss | 10000 | 9900 | 0.000000 | 1.000000 | 0.753472 | 0.985496 | 0.777629 | 1.007462 | 0.024157 | 0.021967 | 1.000000 | 14.754200 | 0.573700 |
| DiffDPO_no_lose_gap | 2000 | 2100 | 0.690165 | 1.000000 | -0.001195 | 0.002517 | 0.066123 | 0.196893 | 0.067318 | 0.194376 | 0.475000 | 24.162500 | 0.865800 |
| DiffDPO_no_lose_gap | 4000 | 3900 | 0.679203 | 1.000000 | -0.005617 | 0.013199 | 0.159720 | 0.873030 | 0.165337 | 0.859831 | 0.200000 | 24.178100 | 0.866200 |
| DiffDPO_no_lose_gap | 6000 | 6000 | 0.692917 | 0.975000 | -0.000092 | 0.000160 | 0.001121 | 0.007014 | 0.001214 | 0.006854 | 0.615385 | 24.189300 | 0.866500 |
| DiffDPO_no_lose_gap | 8000 | 8100 | 0.693014 | 1.000000 | -0.000053 | 0.000116 | 0.000409 | 0.003717 | 0.000462 | 0.003601 | 0.662500 | 24.194600 | 0.866600 |
| DiffDPO_no_lose_gap | 10000 | 9900 | 0.689204 | 1.000000 | -0.001581 | 0.003557 | 0.005808 | 0.034778 | 0.007389 | 0.031221 | 0.312500 | 24.193600 | 0.866500 |

Validation failure 记录：

| experiment | step | error |
| --- | --- | --- |
| VideoDPO_on_VideoInpainting_data | 2000 | ModuleNotFoundError: No module named 'skimage' |
| VideoDPO_on_VideoInpainting_data | 4000 | ModuleNotFoundError: No module named 'skimage' |
| VideoDPO_on_VideoInpainting_data | 6000 | ModuleNotFoundError: No module named 'skimage' |

## 8. VideoDPO 300-Step Loss and Metric Changes

开源 VideoDPO 每一步都有 `[dpo_diag]`，主图和这里的表格都按 300 step 聚合，避免逐 step outlier 把 y 轴拉爆。第四个实验现在同时包含每 300 step 的 `[dpo_diag]` 和更密的 Lightning 进度条指标；这里也按同一粒度聚合展示：

| experiment | step_range | l_dpo | implicit_acc | win_gap | lose_gap | mse_w | ref_mse_w | mse_l | ref_mse_l | loser_dominant_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VideoDPO_open_source | 0-299 | 1.206316 | 0.497656 | 0.000145 | 0.000100 | 0.061527 | 0.061382 | 0.051425 | 0.051325 | 0.659979 |
| VideoDPO_open_source | 300-599 | 1.126580 | 0.521094 | 0.000167 | 0.000247 | 0.054141 | 0.053975 | 0.060886 | 0.060639 | 0.681625 |
| VideoDPO_open_source | 600-899 | 1.295046 | 0.524349 | 0.000171 | 0.000160 | 0.059590 | 0.059419 | 0.052617 | 0.052456 | 0.677229 |
| VideoDPO_open_source | 900-1199 | 1.230712 | 0.538307 | 0.000216 | 0.000359 | 0.066572 | 0.066356 | 0.056633 | 0.056274 | 0.739676 |
| VideoDPO_open_source | 1200-1499 | 1.117210 | 0.541354 | 0.000278 | 0.000373 | 0.045000 | 0.044723 | 0.052992 | 0.052619 | 0.779135 |
| VideoDPO_open_source | 1500-1799 | 1.324093 | 0.534896 | 0.000411 | 0.000506 | 0.059032 | 0.058620 | 0.063555 | 0.063049 | 0.820704 |
| VideoDPO_open_source | 1800-2099 | 1.230840 | 0.541510 | 0.000388 | 0.000505 | 0.054248 | 0.053860 | 0.061222 | 0.060717 | 0.830485 |
| VideoDPO_open_source | 2100-2399 | 1.156703 | 0.561875 | 0.000551 | 0.000730 | 0.050740 | 0.050190 | 0.050605 | 0.049876 | 0.882955 |
| VideoDPO_open_source | 2400-2699 | 1.228116 | 0.546172 | 0.000748 | 0.000863 | 0.056884 | 0.056135 | 0.052842 | 0.051978 | 0.910859 |
| VideoDPO_open_source | 2700-2999 | 1.248778 | 0.569818 | 0.000866 | 0.001028 | 0.056111 | 0.055245 | 0.065309 | 0.064282 | 0.931438 |
| VideoDPO_open_source | 3000-3299 | 1.176092 | 0.571016 | 0.001095 | 0.001406 | 0.062999 | 0.061904 | 0.061665 | 0.060259 | 0.952514 |
| VideoDPO_open_source | 3300-3599 | 1.261222 | 0.581328 | 0.001064 | 0.001329 | 0.060322 | 0.059258 | 0.054577 | 0.053248 | 0.939504 |
| VideoDPO_open_source | 3600-3899 | 1.321320 | 0.571458 | 0.001115 | 0.001366 | 0.062826 | 0.061711 | 0.059920 | 0.058554 | 0.954483 |
| VideoDPO_open_source | 3900-4199 | 1.258199 | 0.583073 | 0.001544 | 0.001821 | 0.060281 | 0.058738 | 0.056017 | 0.054196 | 0.966241 |
| VideoDPO_open_source | 4200-4499 | 1.103188 | 0.605651 | 0.001535 | 0.001962 | 0.044227 | 0.042692 | 0.060070 | 0.058108 | 0.974421 |
| VideoDPO_open_source | 4500-4799 | 1.369577 | 0.616276 | 0.001877 | 0.002295 | 0.072283 | 0.070406 | 0.063451 | 0.061155 | 0.975902 |
| VideoDPO_open_source | 4800-5099 | 1.186411 | 0.614688 | 0.001697 | 0.002205 | 0.058371 | 0.056675 | 0.056770 | 0.054565 | 0.980566 |
| VideoDPO_open_source | 5100-5399 | 1.213998 | 0.644010 | 0.001874 | 0.002387 | 0.060435 | 0.058561 | 0.058478 | 0.056091 | 0.977658 |
| VideoDPO_open_source | 5400-5699 | 0.998974 | 0.646276 | 0.001717 | 0.002334 | 0.054009 | 0.052292 | 0.054677 | 0.052342 | 0.979946 |
| VideoDPO_open_source | 5700-5999 | 1.093623 | 0.631432 | 0.001601 | 0.002157 | 0.060544 | 0.058943 | 0.059400 | 0.057243 | 0.981852 |
| VideoDPO_open_source | 6000-6299 | 1.194956 | 0.643880 | 0.002015 | 0.002624 | 0.062474 | 0.060459 | 0.062389 | 0.059764 | 0.984395 |
| VideoDPO_open_source | 6300-6599 | 1.202682 | 0.646510 | 0.001863 | 0.002387 | 0.055736 | 0.053873 | 0.055176 | 0.052789 | 0.975779 |
| VideoDPO_open_source | 6600-6899 | 1.031278 | 0.665260 | 0.002069 | 0.002920 | 0.050502 | 0.048433 | 0.069063 | 0.066143 | 0.988104 |
| VideoDPO_open_source | 6900-7199 | 1.228688 | 0.639375 | 0.002029 | 0.002627 | 0.055766 | 0.053737 | 0.064628 | 0.062001 | 0.990305 |
| VideoDPO_open_source | 7200-7499 | 1.124569 | 0.649245 | 0.001908 | 0.002631 | 0.058385 | 0.056477 | 0.061662 | 0.059030 | 0.982196 |
| VideoDPO_open_source | 7500-7799 | 1.037432 | 0.659922 | 0.002050 | 0.002766 | 0.055029 | 0.052979 | 0.056852 | 0.054086 | 0.989297 |
| VideoDPO_open_source | 7800-8099 | 1.281346 | 0.656979 | 0.002163 | 0.002786 | 0.056735 | 0.054572 | 0.060926 | 0.058140 | 0.987916 |
| VideoDPO_open_source | 8100-8399 | 1.014541 | 0.704896 | 0.002260 | 0.003102 | 0.064431 | 0.062171 | 0.060019 | 0.056917 | 0.991095 |
| VideoDPO_open_source | 8400-8699 | 1.249460 | 0.684974 | 0.002484 | 0.003204 | 0.060008 | 0.057524 | 0.061672 | 0.058468 | 0.991864 |
| VideoDPO_open_source | 8700-8999 | 1.137975 | 0.694427 | 0.002798 | 0.003700 | 0.069833 | 0.067035 | 0.068081 | 0.064381 | 0.993921 |
| VideoDPO_open_source | 9000-9299 | 1.090771 | 0.676901 | 0.002798 | 0.003727 | 0.060293 | 0.057495 | 0.048876 | 0.045149 | 0.992126 |
| VideoDPO_open_source | 9300-9599 | 1.182224 | 0.693281 | 0.003023 | 0.003707 | 0.056218 | 0.053195 | 0.055183 | 0.051476 | 0.995009 |
| VideoDPO_open_source | 9600-9899 | 0.988804 | 0.696510 | 0.002471 | 0.003357 | 0.058933 | 0.056463 | 0.049648 | 0.046291 | 0.988235 |
| VideoDPO_open_source | 9900-9999 | 1.273232 | 0.649375 | 0.002854 | 0.003358 | 0.049453 | 0.046599 | 0.059372 | 0.056015 | 0.993466 |
| VideoDPO_on_VideoInpainting_data | 0-299 | 0.370394 | 0.866620 | 0.000102 | 0.001299 | 0.015590 | 0.015586 | 0.029767 | 0.029772 | 0.666667 |
| VideoDPO_on_VideoInpainting_data | 300-599 | 0.194907 | 0.915524 | 0.000838 | 0.003323 | 0.082075 | 0.081651 | 0.341015 | 0.337515 | 1.000000 |
| VideoDPO_on_VideoInpainting_data | 600-899 | 0.142905 | 0.928307 | 0.001401 | 0.004295 | 0.079115 | 0.078530 | 0.078880 | 0.076729 | 1.000000 |
| VideoDPO_on_VideoInpainting_data | 900-1199 | 0.214551 | 0.907743 | 0.003649 | 0.007893 | 0.041541 | 0.040519 | 0.041232 | 0.038807 | 1.000000 |
| VideoDPO_on_VideoInpainting_data | 1200-1499 | 0.196145 | 0.931068 | 0.005122 | 0.010162 | 0.086792 | 0.082328 | 0.013563 | 0.006318 | 1.000000 |
| VideoDPO_on_VideoInpainting_data | 1500-1799 | 0.097418 | 0.944407 | 0.004813 | 0.010522 | 0.021182 | 0.011998 | 0.061778 | 0.042346 | 1.000000 |
| VideoDPO_on_VideoInpainting_data | 1800-2099 | 0.122843 | 0.954414 | 0.005659 | 0.011095 | 0.026534 | 0.020604 | 0.019344 | 0.012689 | 1.000000 |
| VideoDPO_on_VideoInpainting_data | 2100-2399 | 0.087726 | 0.956639 | 0.006868 | 0.014825 | 0.021577 | 0.014989 | 0.012566 | 0.003264 | 1.000000 |
| VideoDPO_on_VideoInpainting_data | 2400-2699 | 0.102463 | 0.962748 | 0.006277 | 0.012720 | 0.034246 | 0.025455 | 0.014674 | 0.006279 | 1.000000 |
| VideoDPO_on_VideoInpainting_data | 2700-2999 | 0.086110 | 0.967193 | 0.005872 | 0.012985 | 0.024846 | 0.018530 | 0.114834 | 0.106243 | 1.000000 |
| VideoDPO_on_VideoInpainting_data | 3000-3299 | 0.075562 | 0.968857 | 0.006116 | 0.012037 | 0.273561 | 0.267682 | 0.022190 | 0.013975 | 1.000000 |
| VideoDPO_on_VideoInpainting_data | 3300-3599 | 0.182137 | 0.949400 | 0.008588 | 0.016497 | 0.049261 | 0.041996 | 0.017304 | 0.010007 | 1.000000 |
| VideoDPO_on_VideoInpainting_data | 3600-3899 | 0.114101 | 0.960517 | 0.009279 | 0.018661 | 0.046181 | 0.033531 | 0.092927 | 0.077909 | 1.000000 |
| VideoDPO_on_VideoInpainting_data | 3900-4199 | 0.116813 | 0.956630 | 0.012102 | 0.022557 | 0.016136 | 0.008138 | 0.057907 | 0.031499 | 1.000000 |
| VideoDPO_on_VideoInpainting_data | 4200-4499 | 0.143945 | 0.957181 | 0.010146 | 0.019794 | 0.175611 | 0.167033 | 0.232719 | 0.215578 | 1.000000 |
| VideoDPO_on_VideoInpainting_data | 4500-4799 | 0.048252 | 0.971637 | 0.012022 | 0.022498 | 0.038502 | 0.026792 | 0.071888 | 0.055144 | 1.000000 |
| VideoDPO_on_VideoInpainting_data | 4800-5099 | 0.108688 | 0.965523 | 0.011734 | 0.022080 | 0.138662 | 0.127555 | 0.174029 | 0.147194 | 1.000000 |
| VideoDPO_on_VideoInpainting_data | 5100-5399 | 0.182087 | 0.966080 | 0.011235 | 0.021407 | 0.014709 | 0.001962 | 0.019323 | 0.006180 | 1.000000 |
| VideoDPO_on_VideoInpainting_data | 5400-5699 | 0.074988 | 0.970523 | 0.011769 | 0.022876 | 0.016723 | 0.001763 | 0.333939 | 0.295152 | 1.000000 |
| VideoDPO_on_VideoInpainting_data | 5700-5999 | 0.330176 | 0.938862 | 0.015589 | 0.027609 | 0.016729 | 0.002980 | 0.017590 | 0.002378 | 1.000000 |
| VideoDPO_on_VideoInpainting_data | 6000-6101 | 0.071968 | 0.964033 | 0.014945 | 0.025360 | 0.023744 | 0.004359 | 0.028337 | 0.009168 | 1.000000 |

## 9. Experiment Interpretation

### 9.1 普通 DiffDPO_loss

最后 20%：

- `implicit_acc=1.000000`
- `win_gap=0.846902`
- `lose_gap=1.061145`
- `lose_gap - win_gap=0.214243`
- `loser_dominant_ratio=1.000000`

这个实验的问题不是指标错了，而是裸 DPO 目标可以被“把 loser 变差更多”轻易满足。PSNR/SSIM 同步下降说明 policy 的真实 inpainting 质量也在退化。

### 9.2 删除 lose_gap 的 ablation

最后 20%：

- `implicit_acc=1.000000`
- `win_gap=-0.000600`
- `lose_gap=0.001330`
- `PSNR/SSIM` 基本稳定在 24.19 / 0.866 附近

虽然 loss 中删除了 loser 项，`ml/mrefl/lose_gap` 仍然可以作为 **monitor-only metrics** 计算。它们的意义是观察 policy 对 loser 的副作用，而不是训练信号。这个 ablation 的目的就是确认 winner branch 是否能单独 work；结果是可以。

### 9.3 开源 VideoDPO

VideoDPO 的 `win_gap` 后期略大于 0，但量级很小，且它的优化目标本来就是最终视频偏好，不是最小化每个 winner 的 epsilon MSE。这个现象说明：

- `win_gap > 0` 不能直接推出最终视频一定不如 ref。
- 但在我们的 DiffuEraser 日志里，`win_gap/lose_gap` 的正向漂移是 1e-1 到 1e0 量级，并且 PSNR/SSIM 真实下降，所以是实质性退化。

### 9.4 VideoDPO 使用 VideoInpainting 数据

主窗口最后 20%：

- `implicit_acc=0.959839`
- `win_gap=0.012829`
- `lose_gap=0.023730`
- `lose_gap - win_gap=0.010901`
- `loser_dominant_ratio=1.000000`

这个实验的重点不是“VideoDPO 代码一定坏了”，而是统一成 VideoInpainting 数据后，VideoDPO 也会很快把偏好目标做成 `lose_gap > win_gap`。而且 `loser_dominant_ratio` 在主窗口内几乎一直为 1，说明排序胜利高度依赖 loser 侧变差。它和普通 DiffDPO 的失败方向是一致的，只是数值尺度和 log 频率不同。

第四个实验的 PSNR/SSIM 没有成功记录，原因不是逻辑上没有验证，而是日志里明确报了：

```text
[video_inpaint_val] step=2000 failed: ModuleNotFoundError: No module named 'skimage'
[video_inpaint_val] step=4000 failed: ModuleNotFoundError: No module named 'skimage'
[video_inpaint_val] step=6000 failed: ModuleNotFoundError: No module named 'skimage'
```

因此第四个实验目前只能用于判断 DPO 中间指标趋势；如果要比较最终 inpainting 质量，需要先修 H20 环境的 `skimage` 或把 metric import 改成项目里已有且不依赖 `skimage` 的实现。

## 10. Why Inpainting Preference Pairs Are Harder

你这个判断是对的：Video Inpainting 的偏好对和主流 T2V VideoDPO 不完全一样。

主流 VideoDPO 常见数据：同一个 prompt 通过 T2V 生成多个候选，winner/loser 可能在外观、构图、运动、对象状态上差异很大。相对排序信号更粗，但更容易被 DPO 区分。

我们的 Video Inpainting 数据：

- winner 多来自 GT。
- loser 来自某个 inpainting 模型输出。
- 背景和非 mask 区域大量相同。
- 真正差异主要在 mask 区域，而且常见问题是 blur、artifact、flicker、temporal inconsistency。

这会带来两个问题：

1. **epsilon MSE 可能被大面积相同区域稀释。** 如果 loss/diagnostic 没有足够 mask-aware 或 temporal-aware，模型会看到一个很弱、很局部的偏好信号。
2. **loser 更容易成为捷径。** 因为 loser 的缺陷可能只在 mask 区域，而全局 DPO 只要求相对差距，模型可以通过把 loser 区域进一步弄差来获得高 `implicit_acc`，而不是真正改善 winner/GT reconstruction。

所以 `implicit_acc` 的难易程度确实和数据集 pair separability 有关，但不只由难易程度决定；还受 mask 覆盖、噪声 timestep、beta、loss 权重、ref/policy 初始距离、MSE 是否能捕捉 flicker 等因素影响。对当前任务，`implicit_acc` 必须和 `win_gap`、`lose_gap`、`loser_dominant_ratio`、PSNR/SSIM 一起看。

## 11. Proposed Regularized Objective

建议下一轮采用你图里的方向：

```text
L_total =
  L_DPO_norm
  + lambda_a * m_w
  + lambda_w * ReLU(m_w - m_w_ref)
  + lambda_g * ReLU(tilde_lose_gap - tilde_win_gap - tau_g)
```

含义：

- 当前四个实验的主表和图只展示 `DPO_loss`，因为现有配置里正则权重为 0，`Total_loss` 基本等于 `DPO_loss`。
- 只有下一轮真正加入正则项之后，`Total_loss` 才需要单独进入图表。
- `L_DPO_norm`: 保留偏好排序，但建议对 gap 做 normalization，避免不同 timestep / mask 面积造成尺度漂移。
- `lambda_a * m_w`: Reg-DPO / SFT-style positive anchor，直接约束 winner/GT 分支不能飘。
- `lambda_w * ReLU(m_w - m_w_ref)`: DPOP/Smaug-style positive protection。只有当 policy 在 winner 上比 ref 更差时才惩罚。
- `lambda_g * ReLU(tilde_lose_gap - tilde_win_gap - tau_g)`: anti-loser-dominance。允许 loser 有一定 margin，但当模型主要靠扩大 loser gap 获胜时惩罚。

### Recommended First Sweep

建议先做保守 sweep：

| run | lambda_a | lambda_w | lambda_g | tau_g | note |
|---|---:|---:|---:|---:|---|
| R1 | 0.01 | 1.0 | 0.0 | - | 只验证 winner anchor 是否稳住 PSNR |
| R2 | 0.01 | 1.0 | 0.1 | 0.05 | 加轻量 anti-loser-dominance |
| R3 | 0.05 | 1.0 | 0.1 | 0.05 | 更强 winner reconstruction |
| R4 | 0.01 | 5.0 | 0.1 | 0.05 | 更强 DPOP-style ref anchor |

### Stop / Save Criteria

不要再用 `implicit_acc` 单独判断成功。建议采用：

- 必须：`PSNR/SSIM` 不低于 no-lose-gap ablation 的稳定水平太多。
- 必须：`win_gap` 不能长期大幅为正。
- 必须：`lose_gap - win_gap` 不能持续放大到普通 DiffDPO 的量级。
- 参考：`implicit_acc` 处在 0.55-0.9 可接受；如果过快到 1.0，要检查是否又出现 loser shortcut。

## 12. Paper Connection

- `Smaug.pdf / DPOP`: 指出普通 DPO 可以在 preferred likelihood 下降的情况下仍然提升相对偏好；对应我们这里的 `win_gap > 0` 风险。解决思路是给 winner/preferred 加 positive protection。
- `Reg-DPO_compressed.pdf`: 指出视频 DPO 容易通过拉大正负样本误差差距获得低 loss，并加入 SFT regularization 稳住正样本；这和普通 DiffDPO 的 PSNR/SSIM collapse 高度一致。
- `Anchored Preference Optimization and Contrastive Revisions.pdf`: 指出 preference objective 缺少绝对锚点，数据 pair 如果不够 contrastive 会导致信用分配混乱；对 inpainting 的局部 mask 差异尤其相关。

## 13. Files

- Phase summary: `/home/hj/Video_inpainting_DPO/PRD/assets/dpo_metric_analysis_20260505/phase_summary.csv`
- Delta summary: `/home/hj/Video_inpainting_DPO/PRD/assets/dpo_metric_analysis_20260505/delta_summary.csv`
- Validation-aligned metrics: `/home/hj/Video_inpainting_DPO/PRD/assets/dpo_metric_analysis_20260505/validation_aligned.csv`
- VideoDPO 300-step chunk metrics: `/home/hj/Video_inpainting_DPO/PRD/assets/dpo_metric_analysis_20260505/videodpo_300step_chunks.csv`
- All parsed diagnostics: `/home/hj/Video_inpainting_DPO/PRD/assets/dpo_metric_analysis_20260505/all_diagnostics.csv`
- Full raw diagnostics before analysis cutoff: `/home/hj/Video_inpainting_DPO/PRD/assets/dpo_metric_analysis_20260505/all_diagnostics_full.csv`
- Experiment scope: `/home/hj/Video_inpainting_DPO/PRD/assets/dpo_metric_analysis_20260505/experiment_scope.csv`
- Validation failures: `/home/hj/Video_inpainting_DPO/PRD/assets/dpo_metric_analysis_20260505/validation_failures.csv`
