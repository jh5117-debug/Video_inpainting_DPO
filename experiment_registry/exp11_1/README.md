# exp11-1

## Data

- win: GT/clean YouTube-VOS frames
- lose: D3 comp loser
- mask/source: D3 partial mask from manifest; D3 YouTube-VOS GT-win comp generated-loser; comp=comp

## How

在 Exp10 上加入 frozen-ref prior、boundary、temporal residual proxy；它不是 ProPainter image-space prior，也不是 RAFT optical-flow warp。

## Why

作为 consistency 方向的 proxy sanity check，不验证 real flow-prior。

## Loss

inside=-0.5*beta*gap, beta=10, lose_gap_weight=0.25, winner_abs=0.05, winner_gap=1.0, margin=0, gap=log_ratio, region=region+proxy_flow_prior_boundary; Exp11-proxy adds frozen-ref prior, boundary proxy, temporal residual proxy. Not real RAFT/ProPainter prior.

## Conclusion

可作为 Exp11-proxy Stage1 结果，不能称 real flow-prior；指标提升保守，diag 仍 loser-dominant。 使用标签 Exp11-proxy，不可写成 real flow-prior consistency。
