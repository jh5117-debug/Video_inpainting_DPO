# Exp35 MiniMax Inference Sensitivity

Status: `MINIMAX_INFERENCE_SENSITIVITY_PASS`

This diagnostic performed no training and did not modify Exp30 outputs.

- Rows: `4` (heldout first, then train).
- Step0 checkpoint: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp30_vor_or_multimodel_minimax/minimax_gate64_adapter_v3_20260627/checkpoints/frozen/checkpoint-0`.
- Temporary perturbed checkpoint: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp35_minimax_flow_dpo_rescue/inference_sensitivity_20260627/temporary_perturbed_checkpoint`.
- Perturb scale: `1.01` over `16` transformer tensors.
- Identity control max full MAE: `0.0`.
- Perturbed mean full MAE: `0.08821829589193357`.
- Perturbed mean mask MAE: `0.15630244233590715`.
- Adapter scale sweep: `NOT_APPLICABLE_NO_LORA_SCOPE_IN_EXP30`.

Codex visual review of the generated strips is required before treating this as final.

## Codex Visual Review

- Reviewed strips: `4/4`.
- Identity control: `4/4` visually identical and hash-identical.
- Perturbed checkpoint: `4/4` measurable nonzero response, but visually subtle.
- Collapse / new artifact: `0/4`.
- Adapter scale sweep: `NOT_APPLICABLE_NO_LORA_SCOPE_IN_EXP30`.

Interpretation: MiniMax inference does use the trained transformer weights. The Exp30 no-change failure is therefore not an inference fallback or ignored checkpoint; it is consistent with very weak utility/update scale and low output sensitivity to tiny parameter movement.
