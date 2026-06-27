# Exp30 MiniMax Gate64 Adapter 10-Step V3
Final status: `MINIMAX_ADAPTER_RECIPE_NOT_READY`.

The Gate64 V3 pool enabled this preregistered MiniMax-only micro gate. The run completed both allowed recipes, `frozen` and `ema`, for 10 optimizer steps with MiniMax flow velocity target `epsilon - z0`. Zero-gap and one-step strict reload passed, but the heldout Step10 outputs did not show quality-positive change.

## Data And Identity
- Train rows: 32.
- Heldout rows: 16 scene-disjoint rows.
- Train manifest SHA256: `1eda205d2dc48714269f30eb390d959549387a778e6438267e6aba087ba14196`.
- Heldout manifest SHA256: `84c231ded930d740bf299b27c2a6b1e95d7decdb3665051371c5df90ae9f2ade`.
- VOR-Eval used: false.
- Long training / RC-FPO: false.

## Recipe Metrics

### ema
- Mean full PSNR delta: `-0.001723360`; win count `7/16`.
- Mean mask PSNR delta: `-0.001850513`; win count `8/16`.
- Mean boundary PSNR delta: `-0.003091523`; win count `8/16`.
- Mean outside PSNR delta: `-0.006032939`; win count `3/16`.
- Mean temporal-diff MAE delta: `0.001298963`.

### frozen
- Mean full PSNR delta: `-0.001136464`; win count `6/16`.
- Mean mask PSNR delta: `-0.001067723`; win count `9/16`.
- Mean boundary PSNR delta: `-0.002820745`; win count `8/16`.
- Mean outside PSNR delta: `-0.006339634`; win count `3/16`.
- Mean temporal-diff MAE delta: `0.001228352`.

## Codex Visual Review
- Review pages opened: 8 combined pages covering both recipes and all 16 heldout rows.
- Visual better: 0/32 recipe-row comparisons.
- Tie / no visible Step10 improvement: 32/32.
- Step10 new visible artifact: 0/32, but local/outside metrics are slightly worse on average.
- Main observation: parameters and checkpoints changed, but rendered heldout videos are effectively identical to Step0; no reliable local defect correction, boundary improvement, or temporal improvement is visible.

## Decision
The MiniMax backend remains technical/plumbing positive but not quality-positive. Because neither recipe improves at least two local/effect metrics and visual better count is 0/16 per recipe, no 30-step or longer MiniMax run is authorized from this prompt.
