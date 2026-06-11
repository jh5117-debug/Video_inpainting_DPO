# exp7a-1

## Data

- win: D2 manifest winner
- lose: D2 comp loser
- mask/source: D2 partial mask from manifest; D2 comp partial-mask task; comp=comp

## How

把任务改成真正 partial-mask inpainting，只训练 Stage1，Stage2 保持 SFT-48000。

## Why

验证只动 spatial Stage1 是否能避免破坏 Stage2 temporal prior。

## Loss

inside=-0.5*beta*gap, beta=10, lose_gap_weight=0.25, winner_abs=0.05, winner_gap=1.0, margin=0, gap=raw_gap, region=full

## Conclusion

比 Stage2 DPO 安全，但早期 val 曾混用 DAVIS/small-D2，结论要带协议 caveat。
