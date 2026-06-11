# exp8a-2

## Data

- win: generated/manifest winner, not GT-clean
- lose: D3 comp loser
- mask/source: D3 partial mask from manifest; D3 YouTube-VOS comp generated-loser; comp=comp

## How

在 exp8a 数据设定下 Stage1 和 Stage2 都做 DPO。

## Why

看 Stage2 temporal DPO 能否补救 generated winner 不干净的问题。

## Loss

inside=-0.5*beta*gap, beta=10, lose_gap_weight=0.25, winner_abs=0.05, winner_gap=1.0, margin=0, gap=raw_gap, region=full

## Conclusion

没有补救，Stage2 不能解决脏 winner 信号；diag 风险延续。
