# DiffuEraser reproduction / SFT / metric setting

- status: `historical_complete_registry_partial`
- short_name: `diffueraser_reproduction_sft`
- task: DiffuEraser SFT/reproduction
- data: DiffuEraser official/SFT assets

## Loss

m_w = policy winner MSE; m_l = policy loser MSE; m_w_ref/m_l_ref are reference MSE.
win_gap = m_w - m_w_ref; lose_gap = m_l - m_l_ref.
inside = -0.5 * beta_dpo * (win_gap - lose_gap_weight * lose_gap).
L_DPO = mean[-logsigmoid(inside)].

Experiment-specific loss: SFT reconstruction MSE, not DPO.

## Conclusion

SFT-48000 is a strong YouTube-VOS-tuned baseline, not a naked base.

## Next Action

Keep as baseline reference; do not call converted_weights_step48000 ordinary base.
