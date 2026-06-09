# Exp8 D3 comp region-loss diagnostic

- status: `PAI_gate_remote_diag_found_eval_pending`
- short_name: `regionloss_diagnostic`
- task: Stage1-only partial-mask inpainting
- data: D3 YouTube-VOS comp generated losers

## Loss

m_w = policy winner MSE; m_l = policy loser MSE; m_w_ref/m_l_ref are reference MSE.
win_gap = m_w - m_w_ref; lose_gap = m_l - m_l_ref.
inside = -0.5 * beta_dpo * (win_gap - lose_gap_weight * lose_gap).
L_DPO = mean[-logsigmoid(inside)].

Experiment-specific loss: Region-loss requested for Exp8; implementation must be validated from code before claiming a final region-loss success. Planned form is region-weighted MSE/DPO over mask, boundary, and outside bands.

## Conclusion

Region-loss diagnostic is valid only if implementation and SFT-48000 baseline are confirmed.

## Next Action

Do not expand D3 until Exp7 smallmask/prior sanity is fixed.
