# Exp42 PAI MiniMax Successful-Removal + Bad-Noise Data Breakthrough

This registry tracks the PAI-side MiniMax data-signal lane. Exp42 mines
official MiniMax successful-removal and failure candidates, builds
success-vs-failure bad-noise states, and only then allows short Stage2-style
SFT/DPO gates.

It is intentionally separated from H20 Exp41 and previous MiniMax training
outputs.

## Current Decision

`MINIMAX_SUCCESSFUL_REMOVAL_POOL_WEAK`

Official PAI mining found real MiniMax successful-removal signal, but the
selected success/failure rows are too source-clustered and label-noisy to
unlock bad-noise v3, Stage2 train/search/shadow data, SFT, or DPO.
