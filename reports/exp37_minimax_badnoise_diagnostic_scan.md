# Exp37 MiniMax Bad-Noise Diagnostic Scan

Status: `MINIMAX_BAD_NOISE_STATES_READY`

- Training launched: false.
- Model update: false.
- Rows: `32` train rows from the LocalDPO-style pool.
- Candidate states per row: `64` (`K_noise=8`, `K_timestep=8`).
- Timesteps: `0.05, 0.15, 0.25, 0.35, 0.5, 0.65, 0.8, 0.95`.
- Total states: `2048`.
- Manifest SHA256: `492210b2cd725faa348adcbafaf37bf82cc6790b4eb0607b9f758047d1c795d4`.
- Mean hard-A/random gradient proxy ratio: `0.570900`.
- Max hard-A/random gradient proxy ratio: `0.813708`.

Selection policy:

- `hard_state_A`: maximum local loser residual with outside sanity filter.
- `hard_state_B`: weakest winner advantage / largest preference violation proxy.
- `hard_state_C`: highest winner-loss risk with outside sanity filter.

This milestone mines states only. It does not train, evaluate a recipe, or unlock 30-step.
