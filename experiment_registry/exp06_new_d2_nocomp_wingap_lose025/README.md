# New Exp6 D2 no-comp winner-gap regularized DPO

- status: `completed_h20_diag_local`
- short_name: `new_d2_nocomp_wingap_lose025`
- task: data-only full-mask bridge; model did not see partial mask
- data: D2 partial-mask no-comp K=4 raw generated losers

## Loss

m_w = policy winner MSE; m_l = policy loser MSE; m_w_ref/m_l_ref are reference MSE.
win_gap = m_w - m_w_ref; lose_gap = m_l - m_l_ref.
inside = -0.5 * beta_dpo * (win_gap - lose_gap_weight * lose_gap).
L_DPO = mean[-logsigmoid(inside)].

Experiment-specific loss: L = -logσ{-0.5 * 10 * (win_gap - 0.25 * lose_gap)} + 0.05 * m_w + ReLU(win_gap). Data change: final_loser = raw_loser, no comp.

## Conclusion

New Exp6 is no-comp + winner anchoring, not plain Exp6; promising qualitative progress but not final.

## Next Action

Use for comp/no-comp story; do not rename as plain Exp6.
