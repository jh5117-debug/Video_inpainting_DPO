# Exp9 D3 no-comp Stage1 target-domain gate

- status: `H20_complete_eval_complete_local_diag`
- short_name: `d3_nocomp_target_gate`
- task: Stage1-only partial-mask inpainting
- data: D3 YouTube-VOS no-comp generated losers

## Loss

m_w = policy winner MSE; m_l = policy loser MSE; m_w_ref/m_l_ref are reference MSE.
win_gap = m_w - m_w_ref; lose_gap = m_l - m_l_ref.
inside = -0.5 * beta_dpo * (win_gap - lose_gap_weight * lose_gap).
L_DPO = mean[-logsigmoid(inside)].

Experiment-specific loss: L = -logσ{-0.5 * 10 * (win_gap - 0.25 * lose_gap)} + 0.05 * m_w + ReLU(win_gap). Data change: final_loser = raw_loser no-comp; Stage1 only.

## Conclusion

Completed, but qualitative review says both baseline and DPO can look poor; use as caution.

## Next Action

Compare only after baseline/prior path audit.
