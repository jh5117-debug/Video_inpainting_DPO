# Exp9 D3 comp Stage1 target-domain gate

- status: `PAI_clean_gate_complete_eval_complete_remote_diag_found`
- short_name: `d3_comp_target_gate`
- task: Stage1-only partial-mask inpainting
- data: D3 YouTube-VOS comp generated losers

## Loss

m_w = policy winner MSE; m_l = policy loser MSE; m_w_ref/m_l_ref are reference MSE.
win_gap = m_w - m_w_ref; lose_gap = m_l - m_l_ref.
inside = -0.5 * beta_dpo * (win_gap - lose_gap_weight * lose_gap).
L_DPO = mean[-logsigmoid(inside)].

Experiment-specific loss: L = -logσ{-0.5 * 10 * (win_gap - 0.25 * lose_gap)} + 0.05 * m_w + ReLU(win_gap). Data source = YouTube-VOS D3 comp; Stage1 only.

## Conclusion

ckpt500 is an early-window candidate; longer DPO degraded. Baseline/prior validity still needs care.

## Next Action

Do not scale D3 until Exp7 smallmask/prior audit passes.
