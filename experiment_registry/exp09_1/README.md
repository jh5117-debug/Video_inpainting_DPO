# exp9-1

## Data

- win: GT/clean YouTube-VOS frames
- lose: D3 comp loser
- mask/source: D3 partial mask from manifest; D3 YouTube-VOS GT-win comp generated-loser; comp=comp

## How

把 raw win/lose gap 改为 log-ratio normalized gap，并 clip loser gap；只训练 Stage1，Stage2 用 SFT。

## Why

解决 raw gap 数值尺度不一致，同时保留 DPO pairwise 解释。

## Loss

inside=-0.5*beta*gap, beta=10, lose_gap_weight=0.25, winner_abs=0.05, winner_gap=1.0, margin=0, gap=log_ratio, region=full; norm_gap=log((m+eps)/(m_ref+eps)), loser gap clipped tau=1.0.

## Conclusion

gap 爆炸明显缓解；DAVIS50 raw6 指标小幅超过 SFT48000 baseline，但 diag 仍 loser-dominant。
