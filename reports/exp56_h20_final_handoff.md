# Exp56-H20 Final Handoff

Status: `EXP56_H20_ONESTEP_MIXED_ONLY`

## Answers

1. Did object-only DPO prevent affected / overlap / boundary regression? No. It removed loser-dominant behavior and preserved outside, but overlap / affected / boundary still regressed beyond the Exp56 gate.
2. Did half-step help? Not enough. It slightly reduced overlap regression versus full R5, but worsened affected, boundary, and object gains.
3. Which H20 R5 cell is best? `R5_Q2_T500_S0`.
4. Does any H20 cell reach one-step PASS? No.
5. Should aggregator consider H20 candidate? Yes, `R5_Q2_T500_S0` as mixed diagnostic only, not as a 10-step unlock.
6. Third evidence? No.

## Best Candidate

`R5_Q2_T500_S0`:

- full PSNR delta: +0.013859
- object PSNR delta: +0.956095
- overlap PSNR delta: -0.153271
- affected PSNR delta: -0.084209
- boundary PSNR delta: -0.047360
- outside PSNR delta: +0.047483
- loser contribution ratio: 0.000000
- visual: 0 better / 2 tie / 2 worse

Conclusion: R5 fixed loser dominance but did not fix transition-region safety. No H20 10-step should run before Exp57 cross-lane aggregation.
