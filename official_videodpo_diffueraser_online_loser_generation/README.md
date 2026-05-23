# official_videodpo_diffueraser_online_loser_generation

Purpose: document online loser generation as future work.

Online loser generation is **not** first priority because it is expensive, stochastic, and tightly couples model generation with training. The first version of this project should use offline generated losers with saved manifests.

Do not implement online generation here until:

- DiffuEraser, ProPainter, CoCoCo, and MiniMax-Remover runtimes are all confirmed.
- Offline fullmask and partialmask experiments are stable.
- The training loop can tolerate generation latency and randomness.

Recommended first online design, when ready:

- Cache generated losers per epoch.
- Save every generated loser and manifest entry.
- Keep deterministic seeds per sample where possible.
- Preserve DPO diagnostics and compare against offline baselines.
