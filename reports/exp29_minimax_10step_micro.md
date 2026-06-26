# Exp29 MiniMax 10-Step Micro Gate

Date: 2026-06-26

Status: `MINIMAX_10STEP_PARETO_MIXED`

MiniMax completed a 10-step technical micro gate using conservative SGD
(`lr=1e-7`) after the fp16 AdamW attempt produced NaNs. The run did not NaN,
saved strict-loadable checkpoints, and produced heldout videos. However, the
heldout visual review showed almost no quality change, so this is not a
scientific positive adapter result.

## Diagnostics

- zero-gap: `MINIMAX_ZERO_GAP_PASSED`
- one-step: `MINIMAX_ONE_STEP_STRICT_RELOAD_PASSED`
- NaN detected after optimizer fix: false
- step10 strict reload missing keys: 0
- step10 strict reload unexpected keys: 0
- step10 parameter delta probe: 1.1061271569642785e-10
- peak VRAM: 44614.22 MiB

## Heldout Review

Heldout samples:

- `davis_hockey`
- `davis_koala`

Visual finding: Step10 is nearly indistinguishable from Step0. There is no
clear new collapse, but there is also no visible improvement.

This supports `MINIMAX_ADAPTER_POSSIBLE_NEEDS_MORE_WORK`, not
`MINIMAX_TRUE_ADAPTER_FEASIBILITY_CONFIRMED`.

