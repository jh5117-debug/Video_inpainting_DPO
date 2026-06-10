# Experiment Lineage: Data / How / Why / Conclusion

This PRD keeps the experiment story in one place. It is intentionally written in plain language so slides, registry entries, and final summaries use the same meanings.

| 实验 | data | how | why | conclusion |
|---|---|---|---|---|
| exp1 | DAVIS / YouTube-VOS 风格修复数据；使用 DiffuEraser official / SFT 权重；没有 winner / loser preference pair。 | 复现 DiffuEraser Stage1 / Stage2 推理、mask 设置和 metric setting；真实 loss 是非 DPO 的 SFT / reconstruction denoising MSE。 | 先固定底座、权重路径和 metric，否则后面 DPO 变好或变坏都无法解释。 | 已完成；SFT-48000 是后续必须对比的强 baseline；video inpainting 用 PSNR / SSIM / LPIPS / Ewarp / VFID / TC，不用 VBench。 |
| exp2 | official VideoDPO winner / rejected pairs；使用 official VC2 / VideoDPO 复现资产。 | 完全按 official VideoDPO pipeline 跑 VC2 model + official VideoDPO training / eval。 | 确认 official VideoDPO pipeline 在当前机器和脚本环境中可复现，是框架 sanity check。 | 已完成定性 + 定量；说明框架和评估脚本本身能跑，下一步可以只换模型而不换官方数据流程。 |
| exp3 | 仍使用 official VideoDPO pairs；不引入 generated loser。 | 在 official VideoDPO full-mask bridge 框架下，把 VC2 adapter / model 替换成 DiffuEraser，验证 loader、adapter、Stage1 / Stage2 权重路径。 | 回答 DiffuEraser 能不能接进 VideoDPO 训练框架，排除“接入失败”这个解释。 | 已完成；后续主要问题来自数据和 objective，而不是 DiffuEraser 接入 VideoDPO 框架失败。 |
| exp4 | win 仍是 VideoDPO win；lose 改成 DiffuEraser full-mask generated video；exp4-data 后续已删除。 | 只换 loser 数据，task 仍按 full-mask video generation / bridge 训练，不是 partial-mask inpainting；原计划沿用 old full-video DPO `L_DPO(500,1)`，无 winner anchor。 | 测试“只把 loser 换成 DiffuEraser 生成样本”是否足够。 | 负结果；full-mask generated loser 质量太差，直接暴露数据方向不可继续，推动后续改成 partial-mask K4 / local damage。 |
| exp5 | D2 comp 早期实验；用 VideoDPO / D2 winner-vs-loser 做 DPO。 | 用 comp 数据做 plain / old DPO，早期 loss 较激进。 | 看 comp 数据能不能直接带来提升。 | dpo-diag 显示 `DPO_SATURATED` + `WIN_GAP_EXPLODED` + `COLLAPSE_RISK`，老 loss 太激进。 |
| newexp5 | D2 comp；加入 winner abs + winner gap anchor，`beta=10`，`lose_gap_weight=0.25`。 | 在 exp5 基础上降低 beta 并加入 winner anchor，避免 winner 相对 reference 漂移。 | 修 exp5 的 raw DPO 失控。 | 视觉比 old exp5 稳，但 dpo-diag 仍有 loser-dominant / collapse 风险。 |
| newexp6 | D2 no-comp；同 newexp5 loss。 | 只把数据从 comp 改成 no-comp，其余 loss / anchor 保持一致。 | 比较 comp / no-comp 对 DPO 行为的影响。 | 作为“视频生成式”诊断更清楚，但不是目标 inpainting 协议；dpo-diag 仍有较强 loser shortcut。 |
| exp7a-1 | D2 partial-mask task；Stage1 DPO + SFT Stage2。 | 真正改成 partial-mask inpainting，只训练 spatial Stage1，Stage2 保留 SFT。 | 验证只动 spatial Stage1 是否更安全。 | dpo-diag 比 Stage2 DPO 稳，但 val 数据曾混用 DAVIS / small-D2，需要谨慎解释。 |
| exp7a-2 | D2 partial-mask task；Stage1 DPO + Stage2 DPO。 | Stage1 和 Stage2 都做 DPO，检查 temporal stage 是否能改善。 | 看 Stage2 temporal DPO 是否补时序或反而放大退化。 | Stage2 DPO 更容易放大退化，dpo-diag 有 collapse / win-gap 风险。 |
| exp8a-1 | D3 YouTube-VOS generated-loser；非 GT-win；DAVIS eval；Stage1 DPO + SFT Stage2。 | 把实验迁移到 target domain，但 winner 仍不够干净。 | 看 target-domain generated-loser 是否能直接工作。 | 负面；winner 不干净导致 DPO 信号偏。 |
| exp8a-2 | 同 exp8a，但 Stage2 也 DPO。 | 在不干净 winner 设定下尝试完整 S1 / S2 DPO。 | 看 temporal DPO 是否能补救 exp8a-1。 | 没有补救，dpo-diag 仍 collapse / loser-dominant。 |
| exp8c-1 | D3 YouTube-VOS GT-win；Stage1 DPO + SFT Stage2。 | 把 winner 换成 GT / clean clip，Stage2 保留 SFT。 | 确认 GT winner 是否缓解紫雾和脏 winner 问题。 | 比 exp8a 合理，但 full-loss 仍不够局部。 |
| exp8c-2 | D3 GT-win；Stage1 DPO + Stage2 DPO。 | GT-win 下跑完整 S1 / S2 DPO。 | 看完整 DPO 是否进一步修复 target-domain inpainting。 | dpo-diag 仍有 win-gap / loser-dominant，不能单靠 Stage2 DPO 解决。 |
| exp9-1 | D3 GT-win；log-ratio normalized gap；Stage1 DPO + SFT Stage2。 | 把 raw gap 改成 normalized log-ratio gap，并 clip loser gap。 | 解决 raw win_gap / lose_gap 数值尺度问题，同时保留 DPO pairwise 解释。 | dpo-diag 明显没有旧实验那种大爆炸，固定 protocol 下有小幅提升。 |
| exp9-2 | Exp9 Stage2 DPO；Stage1 DPO + Stage2 DPO。 | 在 normalized-gap 基础上继续训练 Stage2。 | 看 normalized gap 是否让 Stage2 更安全。 | 指标继续略升，但 dpo-diag 仍有 loser-dominant，需要结合定性判断。 |
| exp10-1 | Exp9 + region-local MSE；Stage1 DPO + SFT Stage2。 | 把 MSE 从 full-frame 改成 mask / boundary / outside 加权区域 MSE。 | 让 DPO 优化关注真实补洞区域和边界，而不是被整帧背景稀释。 | SSIM 表现较好，但仍有 collapse-risk 标记。 |
| exp10-2 | Exp10 Stage2 DPO；Stage1 DPO + Stage2 DPO。 | region-local objective 下完整训练 S1 / S2。 | 看 region-local 是否支持完整 spatial + temporal DPO。 | 当前固定 DAVIS50 frame-wise protocol 中 PSNR 最好，但仍必须结合 dpo-diag 和定性，不应只看单个指标。 |
| exp11-1 | Exp10 + flow / prior / boundary consistency proxy；Stage1 DPO + SFT Stage2。 | 在 region-local normalized DPO 上加入当前实现版本的 consistency 项。 | 尝试压紫雾、贴片、边界不连续和 temporal flicker。 | 提升保守，SSIM 没明显超过 baseline；flow / prior 仍是当前实现版本，不应过度宣称。 |
| exp11-2 | Exp11 Stage2 DPO；Stage1 DPO + Stage2 DPO。 | 在 consistency proxy 基础上继续训练 Stage2。 | 测试完整 consistency DPO 是否优于只动 Stage1。 | 当前 SSIM 最好，采样定性没有明显大块紫雾，但仍需要按固定 protocol 和 dpo-diag 共同报告。 |

## Notes

- Exp1-4 are historical setup / bridge / data-gate experiments, not final target-domain DPO results.
- Exp5 onward are DPO lineage experiments and require dpo-diag evidence.
- For video inpainting conclusions, qualitative videos alone are not sufficient; use qualitative review + DAVIS metrics + dpo-diag together.
