# Exp4 fullmask generated loser quality gate

- status: `failed_quality_gate_deleted_artifact`
- short_name: `fullmask_loser_failed_gate`
- task: planned data-only full-mask bridge; no official useful training
- data: fullmask DiffuEraser generated losers from VideoDPO winners

## Loss

m_w = policy winner MSE; m_l = policy loser MSE; m_w_ref/m_l_ref are reference MSE.
win_gap = m_w - m_w_ref; lose_gap = m_l - m_l_ref.
inside = -0.5 * beta_dpo * (win_gap - lose_gap_weight * lose_gap).
L_DPO = mean[-logsigmoid(inside)].

Experiment-specific loss: No reliable DPO training artifact; quality gate failed before formal DPO.

## Conclusion

Generated fullmask losers were too poor; stopped as data negative evidence.

## Next Action

Do not continue Exp4; explain why partial-mask K=4 was introduced.
