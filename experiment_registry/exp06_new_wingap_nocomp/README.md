# newexp6

## Data

- win: D2 manifest winner
- lose: D2 raw no-comp loser
- mask/source: D2 mask; training full-mask bridge; D2 VideoDPO no-comp generated losers; comp=no-comp

## How

loss 与 newexp5 相同，只把 loser 数据从 comp 改成 no-comp。

## Why

做 comp vs no-comp ablation，判断是否 comp 导致视觉/diag 行为差异。

## Loss

inside=-0.5*beta*gap, beta=10, lose_gap_weight=0.25, winner_abs=0.05, winner_gap=1.0, margin=0, gap=raw_gap, region=full

## Conclusion

作为 video-generation 风格诊断更清楚，但不是最终 inpainting evidence；diag 仍有 loser shortcut。
