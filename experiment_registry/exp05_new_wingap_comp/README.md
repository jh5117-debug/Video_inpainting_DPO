# newexp5

## Data

- win: D2 manifest winner
- lose: D2 selected-primary comp loser
- mask/source: D2 mask; training still full-mask bridge; D2 VideoDPO comp generated losers; comp=comp

## How

保留 D2 comp 数据，把 loss 改为 beta=10、lose_gap_weight=0.25，并加入 winner abs / winner gap anchor。

## Why

修复 exp5 中 raw DPO 过激、winner 相对 reference 漂移的问题。

## Loss

inside=-0.5*beta*gap, beta=10, lose_gap_weight=0.25, winner_abs=0.05, winner_gap=1.0, margin=0, gap=raw_gap, region=full

## Conclusion

比 exp5 视觉更稳，但仍非目标 inpainting 协议，diag 仍提示 loser-dominant / collapse 风险。
