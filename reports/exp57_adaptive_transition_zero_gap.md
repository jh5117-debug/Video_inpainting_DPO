# Exp57 Adaptive Transition Zero-Gap

Status: `EXP57_ADAPTIVE_ZERO_GAP_PASS`

Run context:

- Machine: H20 `instance-afs92r3e`
- GPU: 0
- Split: Q2/T500 train4 + heldout4
- Mode: zero-gap / adaptive forward sanity
- Optimizer step: no
- 10-step: no
- VOR-Eval: no
- Hard comp: no

## Checks

| check | result |
| --- | --- |
| preference forward | pass |
| zero-gap margin | pass |
| DPO loss near log(2) | pass (`0.693147`) |
| reference/policy winner loss equality | pass |
| reference/policy loser loss equality | pass |
| adaptive safe-lambda computed | pass |
| transition safety diagnostic computed | pass |
| heldout forward finite | pass |
| NaN/Inf | none observed |
| optimizer step | not run |

## Diagnostics

- safe lambda global: `0.0469255468`
- gradient dot: `10.195175`
- gradient cosine: `0.187702`
- winner grad norm: `7.549371`
- loser grad norm: `7.194732`
- train preference loss: `0.693147`
- heldout preference loss: `0.693147`
- train preference margin: `0.0`
- heldout preference margin: `0.0`
- peak VRAM allocated: `20.016838 GiB`
- runtime: `357.11 sec`

## Decision

The adaptive transition-safe objective passes zero-gap sanity. This unlocks one-step H20/PAI lane execution, but does not unlock 10-step.
