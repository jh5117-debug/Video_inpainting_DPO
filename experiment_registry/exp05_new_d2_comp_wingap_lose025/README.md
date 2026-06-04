# New Exp5 D2 comp winner-gap regularized DPO

- status: `completed_improved_remote_diag_found_not_final`
- short_name: `new_d2_comp_wingap_lose025`
- task: data-only full-mask bridge; model did not see partial mask
- data: D2 partial-mask comp K=4 generated losers

## Loss

m_w = policy winner MSE; m_l = policy loser MSE; m_w_ref/m_l_ref are reference MSE.
win_gap = m_w - m_w_ref; lose_gap = m_l - m_l_ref.
inside = -0.5 * beta_dpo * (win_gap - lose_gap_weight * lose_gap).
L_DPO = mean[-logsigmoid(inside)].

Experiment-specific loss: L = -logσ{-0.5 * 10 * (win_gap - 0.25 * lose_gap)} + 0.05 * m_w + ReLU(win_gap).

## Conclusion

Winner anchoring improved visual quality relative to Old Exp5 collapse but did not solve task mismatch.

## Next Action

Use as guardrail evidence; do not call final success.
