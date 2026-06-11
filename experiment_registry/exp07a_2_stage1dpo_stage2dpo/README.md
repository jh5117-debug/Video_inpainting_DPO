# exp7a-2

## Data

- win: D2 manifest winner
- lose: D2 comp loser
- mask/source: D2 partial mask from manifest; D2 comp partial-mask task; comp=comp

## How

在 exp7a-1 基础上继续对 Stage2 做 DPO。

## Why

检验 temporal Stage2 DPO 是否改善时序，或放大退化。

## Loss

inside=-0.5*beta*gap, beta=10, lose_gap_weight=0.25, winner_abs=0.05, winner_gap=1.0, margin=0, gap=raw_gap, region=full

## Conclusion

Stage2 DPO 更容易放大退化，diag 有 collapse / win-gap 风险。
