# Exp30 MiniMax Gate64 Adapter Zero-Gap V3
Status: `MINIMAX_ZERO_GAP_PASSED` for both frozen-reference and EMA recipes.
- Policy and reference were initialized from the same MiniMax checkpoint clone.
- Target: MiniMax flow velocity `epsilon - z0`.
- Train manifest SHA256: `1eda205d2dc48714269f30eb390d959549387a778e6438267e6aba087ba14196`.
- VOR-Eval used: false.

## ema
- Winner policy/reference loss: `0.542675197` / `0.542675197`.
- Loser policy/reference loss: `0.557255268` / `0.557255268`.
- Win gap: `0.0`; lose gap: `0.0`; DPO loss: `0.6931471824645996`.
- Status: `MINIMAX_ZERO_GAP_PASSED`.

## frozen
- Winner policy/reference loss: `0.542675197` / `0.542675197`.
- Loser policy/reference loss: `0.557255268` / `0.557255268`.
- Win gap: `0.0`; lose gap: `0.0`; DPO loss: `0.6931471824645996`.
- Status: `MINIMAX_ZERO_GAP_PASSED`.
