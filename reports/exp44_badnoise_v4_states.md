# Exp44 Bad-Noise V4 States

- Status: MINIMAX_BADNOISE_V4_READY
- States: 40
- Usable H states: 26
- Minimum gate: 24
- Median local/random gradient-proxy ratio: 2.280567
- Median outside risk vs random: 0.342387

## Interpretation

The state pool is MiniMax-native and same-source: each row keeps the success seed/noise, failure seed/noise, scheduler metadata, GT winner identity, pseudo-success identity, loser identity, mask path, and residual proxy diagnostics.

The gradient proxy is derived from local and outside residuals, not from an optimizer backward pass. This milestone builds data/state metadata only.
