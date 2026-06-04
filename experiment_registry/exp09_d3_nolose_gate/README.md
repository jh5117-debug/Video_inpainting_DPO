# Exp9 D3 comp no-lose-gap gate

- status: `H20_gate_complete_eval_pending_local_diag`
- short_name: `d3_nolose_gate`
- task: Stage1-only partial-mask inpainting
- data: D3 YouTube-VOS comp generated losers

## Loss

m_w = policy winner MSE; m_l = policy loser MSE; m_w_ref/m_l_ref are reference MSE.
win_gap = m_w - m_w_ref; lose_gap = m_l - m_l_ref.
inside = -0.5 * beta_dpo * (win_gap - lose_gap_weight * lose_gap).
L_DPO = mean[-logsigmoid(inside)].

Experiment-specific loss: L = -logσ{-0.5 * 10 * win_gap} + 0.05 * m_w + ReLU(win_gap).

## Conclusion

Tests removing loser-degradation incentive; do not evaluate as final until DAVIS/metric and prior baseline are fixed.

## Next Action

Hold D3 expansion until Exp7 fix.
