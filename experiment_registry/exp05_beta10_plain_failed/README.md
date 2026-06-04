# Exp5 beta10 plain D2 comp DPO

- status: `failed_collapse_remote_diag_found`
- short_name: `beta10_plain_failed`
- task: data-only full-mask bridge; model did not see partial mask
- data: D2 partial-mask comp K=4 generated losers

## Loss

m_w = policy winner MSE; m_l = policy loser MSE; m_w_ref/m_l_ref are reference MSE.
win_gap = m_w - m_w_ref; lose_gap = m_l - m_l_ref.
inside = -0.5 * beta_dpo * (win_gap - lose_gap_weight * lose_gap).
L_DPO = mean[-logsigmoid(inside)].

Experiment-specific loss: L = -logσ{-0.5 * 10 * (win_gap - lose_gap)}; no winner_abs, no winner_gap regularizer.

## Conclusion

Lower beta alone did not solve collapse; missing winner preservation was the core issue.

## Next Action

Use as bridge from Old Exp5 to New Exp5 winner-anchor.
