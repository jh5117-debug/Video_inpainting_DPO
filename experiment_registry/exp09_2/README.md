# exp9-2

## Data

- win: GT/clean YouTube-VOS frames
- lose: D3 comp loser
- mask/source: D3 partial mask from manifest; D3 YouTube-VOS GT-win comp generated-loser; comp=comp

## How

在 log-ratio normalized gap 上继续训练 Stage2。

## Why

检验 normalized gap 是否让 Stage2 DPO 更安全。

## Loss

inside=-0.5*beta*gap, beta=10, lose_gap_weight=0.25, winner_abs=0.05, winner_gap=1.0, margin=0, gap=log_ratio, region=full; norm_gap=log((m+eps)/(m_ref+eps)), loser gap clipped tau=1.0.

## Conclusion

指标仍略优于 baseline，但 Stage2 没有明显放大收益；diag 仍 loser-dominant。
