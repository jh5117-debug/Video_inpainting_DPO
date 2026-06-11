# exp8c-2

## Data

- win: GT/clean YouTube-VOS frames
- lose: D3 comp loser
- mask/source: D3 partial mask from manifest; D3 YouTube-VOS GT-win comp generated-loser; comp=comp

## How

GT-win 设定下 Stage1 和 Stage2 都做 DPO。

## Why

看完整 S1/S2 DPO 是否进一步修复 target-domain inpainting。

## Loss

inside=-0.5*beta*gap, beta=10, lose_gap_weight=0.25, winner_abs=0.05, winner_gap=1.0, margin=0, gap=raw_gap, region=full

## Conclusion

Stage2 DPO 不能单独解决紫雾/贴片风险；diag 仍提示 loser-dominant。
