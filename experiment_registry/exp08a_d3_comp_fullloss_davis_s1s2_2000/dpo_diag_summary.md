# DPO Diagnostics Summary: exp08a

- short_name: `d3_comp_fullloss_davis_s1s2_2000`
- diag_file_status: `REMOTE_DIAG_PATH_FOUND`
- diag_status: `COMPLETE_FROM_USER_PAI_AUDIT`
- stage1_dpo_diag_csv: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260605_142442_exp08_d3_comp_fullloss_wingap_lose025_s1_2000_davis_pai/dpo_diagnostics.csv`
- stage2_dpo_diag_csv: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260606_070556_exp08_d3_comp_fullloss_wingap_lose025_s2_2000_davis_pai/dpo_diagnostics.csv`
- stage1_row_count: `201`
- stage2_row_count: `201`
- last_step: `2000`

## Key Risk Fields

- Stage1 final dpo_loss: `0.164854`
- Stage1 final implicit_acc: `1.000000`
- Stage1 final loser_dominant_ratio: `1.000000`
- Stage1 final mse_l_over_ref_mse_l: `300.585236`
- Stage1 final sigma_term: `0.869146`
- Stage1 final kl_divergence: `0.379077`
- Stage2 final dpo_loss: `0.595376`
- Stage2 final implicit_acc: `1.000000`
- Stage2 final loser_dominant_ratio: `1.000000`
- Stage2 final mse_l_over_ref_mse_l: `51.705112`
- Stage2 final sigma_term: `0.561856`
- Stage2 final kl_divergence: `0.050694`

Interpretation: Exp8a shows a loser-degradation shortcut. The DPO objective reaches high implicit accuracy largely by increasing loser error relative to the reference, while DAVIS metrics for the policy outputs are much worse than the SFT-48000 DiffuEraser baseline.
