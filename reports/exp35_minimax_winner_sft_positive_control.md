# Exp35 MiniMax Winner-SFT Positive-Control

Status: `MINIMAX_POSITIVE_CONTROL_PASS`

- Scope: `S0_current_full_transformer`
- Recipes: `S0_adamw_lr1em05, S0_adamw_lr3em05, S0_adamw_lr0.0001`
- Steps per recipe: `10`
- Training type: winner reconstruction SFT positive-control, not DPO.

Recipe summaries:
- `S0_adamw_lr1em05`: loss decrease `0.6964508928358555`, delta probe `1.4444718289041703e-05`, heldout mask delta `-0.24483777910946536`
- `S0_adamw_lr3em05`: loss decrease `0.6910926904529333`, delta probe `4.1587465275938484e-05`, heldout mask delta `-0.8897026274277025`
- `S0_adamw_lr0.0001`: loss decrease `0.5987974852323532`, delta probe `0.0002197052692736179`, heldout mask delta `-4.2619560566344346`

Codex visual review:

- Opened `12/12` heldout 16-frame `step0|step10|diff` strips.
- `lr=1e-5`: most stable recipe, but `3/4` rows were slightly worse and
  `1/4` was a tie; no clear quality-positive row.
- `lr=3e-5`: visible outside/texture drift increased; `3/4` rows were clearly
  worse and `1/4` was Pareto-mixed/tie.
- `lr=1e-4`: `4/4` rows showed new artifacts, including green/purple color
  drift, black/cyan blocks, blur/occlusion-like failures, and broad outside
  damage.

Interpretation:

This milestone is a positive control for trainability and checkpoint
sensitivity only. It proves MiniMax can be updated and that heldout outputs
change, but the heldout quality is negative under winner-SFT after 10 steps.
Do not advance directly to 30-step or claim third-backbone quality-positive
evidence from this result.
