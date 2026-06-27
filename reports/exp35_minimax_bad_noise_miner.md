# Exp35 MiniMax Bad-Noise / Hard-Timestep Miner

Status: `MINIMAX_BAD_NOISE_STATES_READY`

- Model update: false.
- Candidate states per row: `16` (`K_noise=4`, `K_timestep=4`).
- Train rows: `32`.
- Heldout rows: `16`.
- Timesteps: `0.15, 0.35, 0.55, 0.75`.
- Train manifest SHA256: `fbadd0d2565c4bb49245931742215c4d074c9834b369342398058b4ed9732047`.
- Heldout manifest SHA256: `947f6c0f660229f1da92cb756ee7e03cda4b2215d1ae8f154999574b590ec1fb`.

Selection policy:

- `hard_state_A`: max loser local residual with outside sanity filter.
- `hard_state_B`: max preference violation / weakest winner advantage.
- `hard_state_C`: max winner-risk with outside sanity filter.
- Training-time preregistration options: H0 fixed A, H1 online K=4 worst valid, H2 50% random + 50% fixed A.

This milestone mines states only. It does not train, does not evaluate a recipe, and does not unlock 30-step.
