# Old Exp5 D2 comp plain DPO beta500

- status: `failed_collapse_remote_diag_found`
- short_name: `old_d2_comp_plain_failed`
- task: data-only full-mask bridge; model did not see partial mask
- data: D2 partial-mask comp K=4 generated losers

## Loss

m_w = policy winner MSE; m_l = policy loser MSE; m_w_ref/m_l_ref are reference MSE.
win_gap = m_w - m_w_ref; lose_gap = m_l - m_l_ref.
inside = -0.5 * beta_dpo * (win_gap - lose_gap_weight * lose_gap).
L_DPO = mean[-logsigmoid(inside)].

Experiment-specific loss: L = -logσ{-0.5 * 500 * (win_gap - lose_gap)}; no winner_abs, no winner_gap regularizer.

## Conclusion

Collapsed: ranking can be satisfied while winner visual quality breaks.

## Next Action

Keep as failed DPO evidence; fetch remote dpo CSV if numeric tables are needed.
