# exp8c-1

## Data

- win: GT/clean YouTube-VOS frames
- lose: D3 comp loser
- mask/source: D3 partial mask from manifest; D3 YouTube-VOS GT-win comp generated-loser; comp=comp

## How

把 D3 winner 从 generated high-score rollout 换成 GT / clean video，Stage2 保留 SFT。

## Why

验证 GT-win 是否缓解紫雾和脏 winner 问题。

## Loss

inside=-0.5*beta*gap, beta=10, lose_gap_weight=0.25, winner_abs=0.05, winner_gap=1.0, margin=0, gap=raw_gap, region=full

## Conclusion

比 exp8a 合理，但 full-loss 不够局部，diag 仍有 loser-dominant / win-gap 风险。
