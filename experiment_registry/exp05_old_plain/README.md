# exp5

## Data

- win: VideoDPO generated winner or clean target from D2 manifest (exact manifest pending)
- lose: D2 selected-primary comp loser
- mask/source: D2 generated partial mask, but training used full-mask bridge; D2 VideoDPO comp generated losers; comp=comp

## How

用 D2 comp loser 做早期 plain DPO；训练仍是 full-mask bridge / generation-style diagnostic，不是真正 partial-mask inpainting。

## Why

验证只引入 comp generated/rejected loser 是否能直接提升 DiffuEraser。

## Loss

inside=-0.5*beta*gap, beta=500, lose_gap_weight=1.0, winner_abs=0, winner_gap=0, margin=0, gap=raw_gap, region=full

## Conclusion

失败但有价值：暴露 raw/plain DPO 会走 loser shortcut，dpo-diag 饱和并有 win-gap 爆炸。
