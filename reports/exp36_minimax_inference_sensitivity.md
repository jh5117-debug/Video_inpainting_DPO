# Exp36 MiniMax Inference Sensitivity

Status: `MINIMAX_INFERENCE_SENSITIVITY_PASS`

This diagnostic performed no training and did not modify Exp30 outputs.

- Rows: `4` (heldout first, then train).
- Step0 checkpoint: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp30_vor_or_multimodel_minimax/minimax_gate64_adapter_v3_20260627/checkpoints/frozen/checkpoint-0`.
- Temporary perturbed checkpoint: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp36_minimax_objective_rescue/sensitivity_20260627/output/temporary_perturbed_checkpoint`.
- Perturb scale: `1.01` over `16` transformer tensors.
- Identity control max full MAE: `0.0`.
- Perturbed mean full MAE: `0.08821829589193357`.
- Perturbed mean mask MAE: `0.15630244233590715`.
- Adapter scale sweep: `NOT_APPLICABLE_NO_LORA_SCOPE_IN_EXP30`.

Codex visual review completed `4/4` strips: identity controls were visually identical, perturbed outputs had subtle nonzero response, and no collapse/new artifact/systematic outside damage was observed. This is sensitivity evidence only, not a quality-positive adapter result.
