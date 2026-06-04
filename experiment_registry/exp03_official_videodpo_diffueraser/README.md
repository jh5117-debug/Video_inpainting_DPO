# official VideoDPO framework with DiffuEraser replacement

- status: `historical_complete_registry_partial_missing_diag`
- short_name: `official_videodpo_diffueraser`
- task: full-mask bridge / video-generation style
- data: official VideoDPO pairs with DiffuEraser model bridge

## Loss

m_w = policy winner MSE; m_l = policy loser MSE; m_w_ref/m_l_ref are reference MSE.
win_gap = m_w - m_w_ref; lose_gap = m_l - m_l_ref.
inside = -0.5 * beta_dpo * (win_gap - lose_gap_weight * lose_gap).
L_DPO = mean[-logsigmoid(inside)].

Experiment-specific loss: Bridge used full-video VideoDPO-style loss; exact historical hyperparameters pending backfill.

## Conclusion

DiffuEraser can be wired into VideoDPO; later failures are objective/data issues.

## Next Action

Backfill artifacts if needed.
