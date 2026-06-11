# exp11-2

## Data

- win: GT/clean YouTube-VOS frames
- lose: D3 comp loser
- mask/source: D3 partial mask from manifest; D3 YouTube-VOS GT-win comp generated-loser; comp=comp

## How

在 Exp11-proxy Stage1 基础上继续训练 Stage2。

## Why

测试 proxy consistency 在完整 S1/S2 下是否稳定。

## Loss

inside=-0.5*beta*gap, beta=10, lose_gap_weight=0.25, winner_abs=0.05, winner_gap=1.0, margin=0, gap=log_ratio, region=region+proxy_flow_prior_boundary; Exp11-proxy adds frozen-ref prior, boundary proxy, temporal residual proxy. Not real RAFT/ProPainter prior.

## Conclusion

可作为 Exp11-proxy Stage2 结果；不是 real flow-prior。指标接近 Exp10，但不能过度宣称。 使用标签 Exp11-proxy，不可写成 real flow-prior consistency。
