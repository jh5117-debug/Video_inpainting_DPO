# Status

status: proxy_complete_real_flow_prior_blocked
updated_at: 2026-06-11

Truth-audit result:

- Existing Exp11 Stage1/Stage2 completed as a proxy run.
- Use label: `Exp11-proxy: frozen-ref prior + boundary + temporal residual proxy DPO`.
- `L_prior` uses frozen-ref epsilon prediction, not prior frames/tensors.
- `L_flow` is an adjacent-frame residual proxy, not optical-flow warp
  consistency.
- Existing Exp11 outputs can be used only as proxy results, not real flow-prior
  method results.

Launcher behavior:

- `exp11_flow_prior_consistency_dpo/scripts/launch_exp11_pai.sh` writes the
  blocked audit and exits nonzero before new training.

Existing audited artifacts:

- Stage1 complete: true.
- Stage2 complete: true.
- Stage1 dpo_diag: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260609_2331_exp11_n16_gpus4_7_scratch_exp11_flow_prior_consistency_dpo_s1_2000_davis_pai/dpo_diagnostics.csv`.
- Stage2 dpo_diag: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260609_2331_exp11_n16_gpus4_7_scratch_exp11_flow_prior_consistency_dpo_s2_2000_davis_pai/dpo_diagnostics.csv`.
- Existing all-metric eval complete: true.
- Strict mask-pixel eval present: false.
