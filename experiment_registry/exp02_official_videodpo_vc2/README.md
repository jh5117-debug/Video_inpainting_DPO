# official VideoDPO / VC2 reproduction

- status: `historical_complete_registry_partial_missing_diag`
- short_name: `official_videodpo_vc2`
- task: official VideoDPO VC2
- data: official VideoDPO pairs

## Loss

m_w = policy winner MSE; m_l = policy loser MSE; m_w_ref/m_l_ref are reference MSE.
win_gap = m_w - m_w_ref; lose_gap = m_l - m_l_ref.
inside = -0.5 * beta_dpo * (win_gap - lose_gap_weight * lose_gap).
L_DPO = mean[-logsigmoid(inside)].

Experiment-specific loss: Official VideoDPO loss; exact historical hyperparameters pending artifact backfill.

## Conclusion

Official pipeline sanity check.

## Next Action

Backfill artifacts if needed for provenance.
