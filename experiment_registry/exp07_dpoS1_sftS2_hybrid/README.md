# Exp7 DPO-S1 + frozen SFT/base-S2 hybrid

- status: `eval_complete_negative`
- short_name: `dpoS1_sftS2_hybrid`
- task: no training; hybrid checkpoint build/eval
- data: D2 partial-mask eval

## Loss

m_w = policy winner MSE; m_l = policy loser MSE; m_w_ref/m_l_ref are reference MSE.
win_gap = m_w - m_w_ref; lose_gap = m_l - m_l_ref.
inside = -0.5 * beta_dpo * (win_gap - lose_gap_weight * lose_gap).
L_DPO = mean[-logsigmoid(inside)].

Experiment-specific loss: No training loss; hybrid combines DPO Stage1 weights with frozen SFT/base Stage2.

## Conclusion

Did not rescue Exp7; confirms Stage2 DPO should remain stopped.

## Next Action

Do not treat as final candidate.
