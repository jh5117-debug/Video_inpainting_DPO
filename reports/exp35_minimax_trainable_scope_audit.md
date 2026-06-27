# Exp35 MiniMax Trainable Scope Audit

Status: `MINIMAX_TRAINABLE_SCOPE_CURRENT_OK`

- Checkpoint: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp30_vor_or_multimodel_minimax/minimax_gate64_adapter_v3_20260627/checkpoints/frozen/checkpoint-0`
- Tensor count: `461`
- Total parameters represented: `1127055424`
- LoRA/adapter tensor count: `0`
- Exp30 trainable scope: `all_transformer_parameters`
- Sensitivity status: `MINIMAX_INFERENCE_SENSITIVITY_PASS`
- Perturbed full/mask MAE means: `0.08821829589193357` / `0.15630244233590715`

Conclusion: current Exp30 MiniMax scope is not too small and is not ignored by inference. It is the full transformer scope. Therefore no expanded LoRA scope is prepared in this milestone; the next bottleneck remains objective/update-scale and hard-state selection.

Top module groups by parameter count:
- `blocks.0`: tensors `15`, numel `36991232`
- `blocks.1`: tensors `15`, numel `36991232`
- `blocks.10`: tensors `15`, numel `36991232`
- `blocks.11`: tensors `15`, numel `36991232`
- `blocks.12`: tensors `15`, numel `36991232`
- `blocks.13`: tensors `15`, numel `36991232`
- `blocks.14`: tensors `15`, numel `36991232`
- `blocks.15`: tensors `15`, numel `36991232`
- `blocks.16`: tensors `15`, numel `36991232`
- `blocks.17`: tensors `15`, numel `36991232`
- `blocks.18`: tensors `15`, numel `36991232`
- `blocks.19`: tensors `15`, numel `36991232`
