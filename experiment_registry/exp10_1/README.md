# exp10-1

## Data

- win: GT/clean YouTube-VOS frames
- lose: D3 comp loser
- mask/source: D3 partial mask from manifest; D3 YouTube-VOS GT-win comp generated-loser; comp=comp

## How

在 Exp9 基础上把 MSE 改成 mask/boundary/outside 分区加权 MSE，并除以 weight sum。

## Why

让 DPO 优化集中在真实补洞区域和边界，而不是被整帧背景稀释。

## Loss

inside=-0.5*beta*gap, beta=10, lose_gap_weight=0.25, winner_abs=0.05, winner_gap=1.0, margin=0, gap=log_ratio, region=region; region MSE=mask 1.0 / boundary 0.5 / outside 0.05, divide by weight sum.

## Conclusion

固定 DAVIS50 raw6 whole-frame/bbox 指标明显提升；diag 仍有 collapse-risk，需严格 mask metric 继续确认。
