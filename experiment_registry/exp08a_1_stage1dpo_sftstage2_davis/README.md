# exp8a-1

## Data

- win: generated/manifest winner, not GT-clean
- lose: D3 comp loser
- mask/source: D3 partial mask from manifest; D3 YouTube-VOS comp generated-loser; comp=comp

## How

迁移到 D3 / YouTube-VOS generated-loser 数据，但 winner 仍是 generated high-score rollout，不是 GT。

## Why

测试 target-domain generated winner/loser pair 能否直接作为 inpainting DPO 信号。

## Loss

inside=-0.5*beta*gap, beta=10, lose_gap_weight=0.25, winner_abs=0.05, winner_gap=1.0, margin=0, gap=raw_gap, region=full

## Conclusion

负面：winner 不够干净，DPO 信号偏；diag 仍 collapse / loser-dominant。
