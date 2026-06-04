# Exp7 current D2 partial-mask task gate

- status: `failed_or_suspicious_remote_diag_found`
- short_name: `partialmask_task_current_failed`
- task: true partial-mask inpainting
- data: D2 comp manifest with generated partial masks

## Loss

m_w = policy winner MSE; m_l = policy loser MSE; m_w_ref/m_l_ref are reference MSE.
win_gap = m_w - m_w_ref; lose_gap = m_l - m_l_ref.
inside = -0.5 * beta_dpo * (win_gap - lose_gap_weight * lose_gap).
L_DPO = mean[-logsigmoid(inside)].

Experiment-specific loss: L = -logσ{-0.5 * 10 * (win_gap - 0.25 * lose_gap)} + 0.05 * m_w + ReLU(win_gap). Task change: TRAIN_MASK_MODE=partial, MASK_FROM_MANIFEST=true.

## Conclusion

Task changed but quality is unstable; base also poor, so eval/prior/domain/mask must be audited before more DPO.

## Next Action

Run Exp7 small-mask + ProPainter-prior fix gate only after data check.
