# Exp7 fix: VideoDPO small-mask 15-20 + ProPainter prior

- status: `planned_registry_ready_data_check_pending`
- short_name: `fix_smallmask_prior`
- task: Stage1-only partial-mask inpainting gate
- data: new non-overwriting VideoDPO small-mask generated losers

## Loss

m_w = policy winner MSE; m_l = policy loser MSE; m_w_ref/m_l_ref are reference MSE.
win_gap = m_w - m_w_ref; lose_gap = m_l - m_l_ref.
inside = -0.5 * beta_dpo * (win_gap - lose_gap_weight * lose_gap).
L_DPO = mean[-logsigmoid(inside)].

Experiment-specific loss: L = -logσ{-0.5 * 10 * (win_gap - 0.25 * lose_gap)} + 0.05 * m_w + ReLU(win_gap). Prior/data fix: small mask 15%-20% and ProPainter prior.

## Conclusion

Planned correction for Exp7 before expanding D3.

## Next Action

Generate/check smallmask data, then H20 Stage1 gate if data is ready.
