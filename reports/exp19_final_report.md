# Exp19 Final PAI Gate Report

## Status

- Exp19b Stage2 boundary-gated flow adapter 500-step gate: **completed**.
- Adapter type: isolated hook-based Stage2 flow adapter.
- Base model: Exp11 outer b0.75 S2 Stage2, frozen.
- Trainable params: Exp19 flow projectors only.
- Flow cache: limit100 ProPainter completed bidirectional flow, forward-backward confidence, no GT confidence.
- DAVIS10 eval: **blocked**, because the existing evaluator cannot pass external flow tensors/context windows into the pipeline UNet and would silently fall back to Exp11 behavior if used directly.

## Paths

- flow cache: `/mnt/nas/hj/H20_Video_inpainting_DPO/data/cache/exp19_propainter_completed_flow_limit100/`
- run dir: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/exp19b_boundary_flow_adapter_s2_500_limit100/`
- last adapter weights: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/exp19b_boundary_flow_adapter_s2_500_limit100/last_weights/flow_adapter.pt`
- dpo diag: `exp19_boundary_gated_flow_adapter_dpo/dpo_diag/exp19b_stage2_500_dpo_diagnostics.csv`
- preflight report: `reports/exp19_isolated_wrapper_preflight.md`
- eval blocker report: `reports/exp19_eval_wrapper_status.md`

## Training Result

- completed steps: `500`
- checkpoint-250 saved: yes
- checkpoint-500 saved: yes
- last_weights saved: yes
- NaN/OOM/Traceback: none observed in training log
- zero-init equality: passed
- base_grad_norm max: `0.0`
- adapter_grad_norm mean: `0.000468085`
- adapter_residual_norm max: `0.21669`
- gate_mean mean: `0.00666145`
- nonzero_gate_ratio mean: `0.0229342`

## Decision

Exp19 is no longer architecture-blocked for training: the isolated wrapper works and adapter-only optimization runs to 500 steps. It is still **evaluation-blocked** until an Exp19 inference wrapper is implemented. That wrapper must compute/reuse completed flow for each DAVIS video, align flow slices with the pipeline context windows, load `flow_adapter.pt`, and pass flow context into the hooked UNet during denoising.

Do not expand to 1000 steps or DAVIS50 yet, because the positive gate requires DAVIS10 metrics and visual judgment.
