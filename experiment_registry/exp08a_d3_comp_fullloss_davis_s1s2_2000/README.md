# Exp8a D3 Comp Full-Loss DAVIS S1/S2 2000

- status: `completed_negative_davis_metrics_and_diag_present`
- short_name: `d3_comp_fullloss_davis_s1s2_2000`
- task: partial-mask video inpainting
- data: D3 YouTube-VOS selected-primary comp generated losers

## Scope

Exp8a is the ordinary full-loss regularized DPO baseline on target-domain D3 comp data. It is not the region-loss experiment. Region-weighted loss remains Exp8b / future ablation.

## Loss

m_w = policy winner MSE; m_l = policy loser MSE; m_w_ref/m_l_ref are reference MSE.
win_gap = m_w - m_w_ref; lose_gap = m_l - m_l_ref.

L_total = -logsigmoid(-0.5 * 10 * (win_gap - 0.25 * lose_gap)) + 0.05 * m_w + ReLU(win_gap).

## Final Evidence From PAI Audit

Current state is from user-pasted PAI audit context. Codex did not execute on PAI.

- Stage1 2000-step training completed.
- Stage1 `checkpoint-2000`, `last_weights`, and `dpo_diagnostics.csv` were reported present.
- Stage1 DAVIS validation completed at `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp08a_fullloss_stage1_val_davis_20260606_070556`.
- Stage2 2000-step training completed.
- Stage2 `checkpoint-2000`, `last_weights`, and `dpo_diagnostics.csv` were reported present.
- Stage2 DAVIS validation completed at `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp08a_fullloss_stage2_val_davis_20260606_070556`.

## Result

Exp8a is complete and negative. `DPO-S1_SFT-S2` and `DPO-S1_DPO-S2` both underperform DiffuEraser-base on DAVIS boundary, mask-region, and whole-video metrics. DPO diagnostics show loser-degradation behavior, especially large `mse_l_over_ref_mse_l`. Do not report this as region loss or success.
