# exp10-2

## Data

- win: GT/clean YouTube-VOS frames
- lose: D3 comp loser
- mask/source: D3 partial mask from manifest; D3 YouTube-VOS GT-win comp generated-loser; comp=comp

## How

region-local objective 下继续训练 Stage2。

## Why

测试 region-local 是否能支撑完整 spatial + temporal DPO。

## Loss

inside=-0.5*beta*gap, beta=10, lose_gap_weight=0.25, winner_abs=0.05, winner_gap=1.0, margin=0, gap=log_ratio, region=region; region MSE=mask 1.0 / boundary 0.5 / outside 0.05, divide by weight sum.

## Conclusion

当前 DAVIS50 raw6 表里 PSNR 最好；但 diag 仍 loser-dominant，不能只看单个指标。
