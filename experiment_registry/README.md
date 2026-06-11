# Experiment Registry

| experiment | stage | conclusion |
| --- | --- | --- |
| exp5 | old Stage1 DPO + old Stage2 DPO | 失败但有价值：暴露 raw/plain DPO 会走 loser shortcut，dpo-diag 饱和并有 win-gap 爆炸。 |
| newexp5 | Stage1 DPO + Stage2 DPO | 比 exp5 视觉更稳，但仍非目标 inpainting 协议，diag 仍提示 loser-dominant / collapse 风险。 |
| newexp6 | Stage1 DPO + Stage2 DPO | 作为 video-generation 风格诊断更清楚，但不是最终 inpainting evidence；diag 仍有 loser shortcut。 |
| exp7a-1 | DPO-S1 + SFT-S2 | 比 Stage2 DPO 安全，但早期 val 曾混用 DAVIS/small-D2，结论要带协议 caveat。 |
| exp7a-2 | DPO-S1 + DPO-S2 | Stage2 DPO 更容易放大退化，diag 有 collapse / win-gap 风险。 |
| exp8a-1 | DPO-S1 + SFT-S2 | 负面：winner 不够干净，DPO 信号偏；diag 仍 collapse / loser-dominant。 |
| exp8a-2 | DPO-S1 + DPO-S2 | 没有补救，Stage2 不能解决脏 winner 信号；diag 风险延续。 |
| exp8c-1 | DPO-S1 + SFT-S2 | 比 exp8a 合理，但 full-loss 不够局部，diag 仍有 loser-dominant / win-gap 风险。 |
| exp8c-2 | DPO-S1 + DPO-S2 | Stage2 DPO 不能单独解决紫雾/贴片风险；diag 仍提示 loser-dominant。 |
| exp9-1 | DPO-S1 + SFT-S2 | gap 爆炸明显缓解；DAVIS50 raw6 指标小幅超过 SFT48000 baseline，但 diag 仍 loser-dominant。 |
| exp9-2 | DPO-S1 + DPO-S2 | 指标仍略优于 baseline，但 Stage2 没有明显放大收益；diag 仍 loser-dominant。 |
| exp10-1 | DPO-S1 + SFT-S2 | 固定 DAVIS50 raw6 whole-frame/bbox 指标明显提升；diag 仍有 collapse-risk，需严格 mask metric 继续确认。 |
| exp10-2 | DPO-S1 + DPO-S2 | 当前 DAVIS50 raw6 表里 PSNR 最好；但 diag 仍 loser-dominant，不能只看单个指标。 |
| exp11-1 | DPO-S1 + SFT-S2 | 可作为 Exp11-proxy Stage1 结果，不能称 real flow-prior；指标提升保守，diag 仍 loser-dominant。 使用标签 Exp11-proxy，不可写成 real flow-prior consistency。 |
| exp11-2 | DPO-S1 + DPO-S2 | 可作为 Exp11-proxy Stage2 结果；不是 real flow-prior。指标接近 Exp10，但不能过度宣称。 使用标签 Exp11-proxy，不可写成 real flow-prior consistency。 |
