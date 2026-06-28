# Exp37 MiniMax LocalDPO-BadNoise 10-Step Recipes

Status: `MINIMAX_LOCALDPO_BADNOISE_PARETO_MIXED`

This milestone ran exactly the preregistered 10-step recipes on the locked
Exp37 LocalDPO-style train32/heldout16 manifests. No 30-step or long training
was launched.

## Configuration

- Recipes: `R1,R2,R3`
- Train rows: `32`
- Heldout rows: `16`
- Linear-DPO steps: `10`
- R3 winner-SFT warmup steps: `5`
- LR: `1e-05`
- Utility scale: `18.0`
- Winner anchor: `0.05`
- Outside preservation: `0.02`
- Primary heldout visual gate: better `>=6/16`, worse/new artifact `<=4/16`
- Training launched: `true`
- 30-step launched: `false`

## Metrics

| Recipe | Full PSNR delta | Mask PSNR delta | Boundary PSNR delta | Outside PSNR delta | Mean pixel diff | NaN/Inf | Reference delta |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| R1 `LocalDPO-Linear-HardNoise` | `+0.200826` | `+0.161946` | `-0.049755` | `+0.028198` | `3.664826` | no | `0.0` |
| R2 `LocalDPO-Linear-SDPO` | `-0.472765` | `-0.188106` | `-0.582170` | `-1.097125` | `5.630061` | no | `0.0` |
| R3 `LocalDPO-SFTWarmup-Linear` | `-0.564346` | `-0.132147` | `-0.602426` | `-1.374199` | `6.245088` | no | `0.0` |

R2 passed the SDPO preflight with mean safe lambda `0.837984`, but the heldout
metrics were still negative. All recipes passed the zero-gap plumbing check and
kept the reference frozen.

## Codex Visual Review

Codex opened and reviewed `48/48` Step0-vs-Step10 heldout temporal strips:

| Recipe | Reviewed | Better | Tie / no useful gain | Worse / new artifact |
| --- | ---: | ---: | ---: | ---: |
| R1 | 16 | 1 | 15 | 0 |
| R2 | 16 | 1 | 15 | 0 |
| R3 | 16 | 1 | 15 | 0 |

The only visibly better row in each recipe was
`REAL_ENV104_00001_001_01`, where Step10 suppressed a bright local residual.
The remaining rows showed either no visible improvement or low-amplitude
background/texture perturbations. R2 and R3 were especially weak because their
aggregate boundary and outside metrics degraded while visual quality did not
improve.

## Decision

The 10-step rescue is not quality-positive:

- Better rows are `1/16` for every recipe, below the `6/16` gate.
- R1 is metric-mixed rather than visually positive.
- R2 and R3 are metric-negative on full, mask, boundary, and outside PSNR.
- Step10 is not visually identical to Step0, so this is no longer a pure
  no-output-change failure.
- The result is a Pareto-mixed objective/data signal, not third-backbone
  adapter evidence.

Final status: `MINIMAX_LOCALDPO_BADNOISE_PARETO_MIXED`.

30-step remains locked because the required 10-step positive gate was not met.
