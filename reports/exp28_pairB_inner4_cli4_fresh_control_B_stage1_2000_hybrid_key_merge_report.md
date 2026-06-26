# DPO-S1 + SFT-S2 Hybrid Key Merge Report

mode: `dpo_spatial_sft_motion`
dry_run: `False`
dpo_stage1_weights: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp28_fine_inner_boundary_sweep/eval_exports/pairB_inner4_cli4/fresh_control_B_stage1_2000_weights`
sft_stage2_weights: `/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000`
output_dir: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp28_fine_inner_boundary_sweep/eval_exports/pairB_inner4_cli4/fresh_control_B_stage1_2000_hybrid_sft_s2`

## Counts

| category | count |
| --- | ---: |
| loaded_from_dpo_stage1 | 688 |
| loaded_from_sft_stage2 | 567 |
| skipped | 2 |
| shape_mismatch | 0 |
| unexpected | 0 |
| missing | 0 |
| uncertain_preserved_from_sft_stage2 | 0 |

## Safety Notes

- DPO Stage1 spatial/appearance modules are copied into the Stage2 motion UNet.
- SFT Stage2 motion/temporal keys are preserved in the motion UNet.
- DPO Stage1 BrushNet is copied as the hybrid BrushNet.
- Uncertain preserved keys are left from SFT Stage2 and listed for audit; they are not silently overwritten by SFT over DPO spatial modules.

## Loaded From DPO Stage1 Sample

- `unet_main.conv_in.weight`
- `unet_main.conv_in.bias`
- `unet_main.time_embedding.linear_1.weight`
- `unet_main.time_embedding.linear_1.bias`
- `unet_main.time_embedding.linear_2.weight`
- `unet_main.time_embedding.linear_2.bias`
- `unet_main.down_blocks.0.resnets.0.norm1.weight`
- `unet_main.down_blocks.0.resnets.0.norm1.bias`
- `unet_main.down_blocks.0.resnets.0.conv1.weight`
- `unet_main.down_blocks.0.resnets.0.conv1.bias`
- `unet_main.down_blocks.0.resnets.0.time_emb_proj.weight`
- `unet_main.down_blocks.0.resnets.0.time_emb_proj.bias`
- `unet_main.down_blocks.0.resnets.0.norm2.weight`
- `unet_main.down_blocks.0.resnets.0.norm2.bias`
- `unet_main.down_blocks.0.resnets.0.conv2.weight`
- `unet_main.down_blocks.0.resnets.0.conv2.bias`
- `unet_main.down_blocks.0.resnets.1.norm1.weight`
- `unet_main.down_blocks.0.resnets.1.norm1.bias`
- `unet_main.down_blocks.0.resnets.1.conv1.weight`
- `unet_main.down_blocks.0.resnets.1.conv1.bias`
- `unet_main.down_blocks.0.resnets.1.time_emb_proj.weight`
- `unet_main.down_blocks.0.resnets.1.time_emb_proj.bias`
- `unet_main.down_blocks.0.resnets.1.norm2.weight`
- `unet_main.down_blocks.0.resnets.1.norm2.bias`
- `unet_main.down_blocks.0.resnets.1.conv2.weight`
- `unet_main.down_blocks.0.resnets.1.conv2.bias`
- `unet_main.down_blocks.0.attentions.0.norm.weight`
- `unet_main.down_blocks.0.attentions.0.norm.bias`
- `unet_main.down_blocks.0.attentions.0.proj_in.weight`
- `unet_main.down_blocks.0.attentions.0.proj_in.bias`
- `unet_main.down_blocks.0.attentions.0.transformer_blocks.0.norm1.weight`
- `unet_main.down_blocks.0.attentions.0.transformer_blocks.0.norm1.bias`
- `unet_main.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight`
- `unet_main.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.weight`
- `unet_main.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v.weight`
- `unet_main.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0.weight`
- `unet_main.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0.bias`
- `unet_main.down_blocks.0.attentions.0.transformer_blocks.0.norm2.weight`
- `unet_main.down_blocks.0.attentions.0.transformer_blocks.0.norm2.bias`
- `unet_main.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.weight`
- `unet_main.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight`
- `unet_main.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_v.weight`
- `unet_main.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_out.0.weight`
- `unet_main.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_out.0.bias`
- `unet_main.down_blocks.0.attentions.0.transformer_blocks.0.norm3.weight`
- `unet_main.down_blocks.0.attentions.0.transformer_blocks.0.norm3.bias`
- `unet_main.down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj.weight`
- `unet_main.down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj.bias`
- `unet_main.down_blocks.0.attentions.0.transformer_blocks.0.ff.net.2.weight`
- `unet_main.down_blocks.0.attentions.0.transformer_blocks.0.ff.net.2.bias`
- `unet_main.down_blocks.0.attentions.0.proj_out.weight`
- `unet_main.down_blocks.0.attentions.0.proj_out.bias`
- `unet_main.down_blocks.0.attentions.1.norm.weight`
- `unet_main.down_blocks.0.attentions.1.norm.bias`
- `unet_main.down_blocks.0.attentions.1.proj_in.weight`
- `unet_main.down_blocks.0.attentions.1.proj_in.bias`
- `unet_main.down_blocks.0.attentions.1.transformer_blocks.0.norm1.weight`
- `unet_main.down_blocks.0.attentions.1.transformer_blocks.0.norm1.bias`
- `unet_main.down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_q.weight`
- `unet_main.down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_k.weight`
- `unet_main.down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_v.weight`
- `unet_main.down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_out.0.weight`
- `unet_main.down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_out.0.bias`
- `unet_main.down_blocks.0.attentions.1.transformer_blocks.0.norm2.weight`
- `unet_main.down_blocks.0.attentions.1.transformer_blocks.0.norm2.bias`
- `unet_main.down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_q.weight`
- `unet_main.down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_k.weight`
- `unet_main.down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_v.weight`
- `unet_main.down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_out.0.weight`
- `unet_main.down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_out.0.bias`
- `unet_main.down_blocks.0.attentions.1.transformer_blocks.0.norm3.weight`
- `unet_main.down_blocks.0.attentions.1.transformer_blocks.0.norm3.bias`
- `unet_main.down_blocks.0.attentions.1.transformer_blocks.0.ff.net.0.proj.weight`
- `unet_main.down_blocks.0.attentions.1.transformer_blocks.0.ff.net.0.proj.bias`
- `unet_main.down_blocks.0.attentions.1.transformer_blocks.0.ff.net.2.weight`
- `unet_main.down_blocks.0.attentions.1.transformer_blocks.0.ff.net.2.bias`
- `unet_main.down_blocks.0.attentions.1.proj_out.weight`
- `unet_main.down_blocks.0.attentions.1.proj_out.bias`
- `unet_main.down_blocks.0.downsamplers.0.conv.weight`
- `unet_main.down_blocks.0.downsamplers.0.conv.bias`
- ... 608 more

## Loaded From SFT Stage2 Motion Sample

- `unet_main.down_blocks.0.motion_modules.0.norm.weight`
- `unet_main.down_blocks.0.motion_modules.0.norm.bias`
- `unet_main.down_blocks.0.motion_modules.0.proj_in.weight`
- `unet_main.down_blocks.0.motion_modules.0.proj_in.bias`
- `unet_main.down_blocks.0.motion_modules.0.transformer_blocks.0.pos_embed.pe`
- `unet_main.down_blocks.0.motion_modules.0.transformer_blocks.0.norm1.weight`
- `unet_main.down_blocks.0.motion_modules.0.transformer_blocks.0.norm1.bias`
- `unet_main.down_blocks.0.motion_modules.0.transformer_blocks.0.attn1.to_q.weight`
- `unet_main.down_blocks.0.motion_modules.0.transformer_blocks.0.attn1.to_k.weight`
- `unet_main.down_blocks.0.motion_modules.0.transformer_blocks.0.attn1.to_v.weight`
- `unet_main.down_blocks.0.motion_modules.0.transformer_blocks.0.attn1.to_out.0.weight`
- `unet_main.down_blocks.0.motion_modules.0.transformer_blocks.0.attn1.to_out.0.bias`
- `unet_main.down_blocks.0.motion_modules.0.transformer_blocks.0.norm2.weight`
- `unet_main.down_blocks.0.motion_modules.0.transformer_blocks.0.norm2.bias`
- `unet_main.down_blocks.0.motion_modules.0.transformer_blocks.0.attn2.to_q.weight`
- `unet_main.down_blocks.0.motion_modules.0.transformer_blocks.0.attn2.to_k.weight`
- `unet_main.down_blocks.0.motion_modules.0.transformer_blocks.0.attn2.to_v.weight`
- `unet_main.down_blocks.0.motion_modules.0.transformer_blocks.0.attn2.to_out.0.weight`
- `unet_main.down_blocks.0.motion_modules.0.transformer_blocks.0.attn2.to_out.0.bias`
- `unet_main.down_blocks.0.motion_modules.0.transformer_blocks.0.norm3.weight`
- `unet_main.down_blocks.0.motion_modules.0.transformer_blocks.0.norm3.bias`
- `unet_main.down_blocks.0.motion_modules.0.transformer_blocks.0.ff.net.0.proj.weight`
- `unet_main.down_blocks.0.motion_modules.0.transformer_blocks.0.ff.net.0.proj.bias`
- `unet_main.down_blocks.0.motion_modules.0.transformer_blocks.0.ff.net.2.weight`
- `unet_main.down_blocks.0.motion_modules.0.transformer_blocks.0.ff.net.2.bias`
- `unet_main.down_blocks.0.motion_modules.0.proj_out.weight`
- `unet_main.down_blocks.0.motion_modules.0.proj_out.bias`
- `unet_main.down_blocks.0.motion_modules.1.norm.weight`
- `unet_main.down_blocks.0.motion_modules.1.norm.bias`
- `unet_main.down_blocks.0.motion_modules.1.proj_in.weight`
- `unet_main.down_blocks.0.motion_modules.1.proj_in.bias`
- `unet_main.down_blocks.0.motion_modules.1.transformer_blocks.0.pos_embed.pe`
- `unet_main.down_blocks.0.motion_modules.1.transformer_blocks.0.norm1.weight`
- `unet_main.down_blocks.0.motion_modules.1.transformer_blocks.0.norm1.bias`
- `unet_main.down_blocks.0.motion_modules.1.transformer_blocks.0.attn1.to_q.weight`
- `unet_main.down_blocks.0.motion_modules.1.transformer_blocks.0.attn1.to_k.weight`
- `unet_main.down_blocks.0.motion_modules.1.transformer_blocks.0.attn1.to_v.weight`
- `unet_main.down_blocks.0.motion_modules.1.transformer_blocks.0.attn1.to_out.0.weight`
- `unet_main.down_blocks.0.motion_modules.1.transformer_blocks.0.attn1.to_out.0.bias`
- `unet_main.down_blocks.0.motion_modules.1.transformer_blocks.0.norm2.weight`
- `unet_main.down_blocks.0.motion_modules.1.transformer_blocks.0.norm2.bias`
- `unet_main.down_blocks.0.motion_modules.1.transformer_blocks.0.attn2.to_q.weight`
- `unet_main.down_blocks.0.motion_modules.1.transformer_blocks.0.attn2.to_k.weight`
- `unet_main.down_blocks.0.motion_modules.1.transformer_blocks.0.attn2.to_v.weight`
- `unet_main.down_blocks.0.motion_modules.1.transformer_blocks.0.attn2.to_out.0.weight`
- `unet_main.down_blocks.0.motion_modules.1.transformer_blocks.0.attn2.to_out.0.bias`
- `unet_main.down_blocks.0.motion_modules.1.transformer_blocks.0.norm3.weight`
- `unet_main.down_blocks.0.motion_modules.1.transformer_blocks.0.norm3.bias`
- `unet_main.down_blocks.0.motion_modules.1.transformer_blocks.0.ff.net.0.proj.weight`
- `unet_main.down_blocks.0.motion_modules.1.transformer_blocks.0.ff.net.0.proj.bias`
- `unet_main.down_blocks.0.motion_modules.1.transformer_blocks.0.ff.net.2.weight`
- `unet_main.down_blocks.0.motion_modules.1.transformer_blocks.0.ff.net.2.bias`
- `unet_main.down_blocks.0.motion_modules.1.proj_out.weight`
- `unet_main.down_blocks.0.motion_modules.1.proj_out.bias`
- `unet_main.down_blocks.1.motion_modules.0.norm.weight`
- `unet_main.down_blocks.1.motion_modules.0.norm.bias`
- `unet_main.down_blocks.1.motion_modules.0.proj_in.weight`
- `unet_main.down_blocks.1.motion_modules.0.proj_in.bias`
- `unet_main.down_blocks.1.motion_modules.0.transformer_blocks.0.pos_embed.pe`
- `unet_main.down_blocks.1.motion_modules.0.transformer_blocks.0.norm1.weight`
- `unet_main.down_blocks.1.motion_modules.0.transformer_blocks.0.norm1.bias`
- `unet_main.down_blocks.1.motion_modules.0.transformer_blocks.0.attn1.to_q.weight`
- `unet_main.down_blocks.1.motion_modules.0.transformer_blocks.0.attn1.to_k.weight`
- `unet_main.down_blocks.1.motion_modules.0.transformer_blocks.0.attn1.to_v.weight`
- `unet_main.down_blocks.1.motion_modules.0.transformer_blocks.0.attn1.to_out.0.weight`
- `unet_main.down_blocks.1.motion_modules.0.transformer_blocks.0.attn1.to_out.0.bias`
- `unet_main.down_blocks.1.motion_modules.0.transformer_blocks.0.norm2.weight`
- `unet_main.down_blocks.1.motion_modules.0.transformer_blocks.0.norm2.bias`
- `unet_main.down_blocks.1.motion_modules.0.transformer_blocks.0.attn2.to_q.weight`
- `unet_main.down_blocks.1.motion_modules.0.transformer_blocks.0.attn2.to_k.weight`
- `unet_main.down_blocks.1.motion_modules.0.transformer_blocks.0.attn2.to_v.weight`
- `unet_main.down_blocks.1.motion_modules.0.transformer_blocks.0.attn2.to_out.0.weight`
- `unet_main.down_blocks.1.motion_modules.0.transformer_blocks.0.attn2.to_out.0.bias`
- `unet_main.down_blocks.1.motion_modules.0.transformer_blocks.0.norm3.weight`
- `unet_main.down_blocks.1.motion_modules.0.transformer_blocks.0.norm3.bias`
- `unet_main.down_blocks.1.motion_modules.0.transformer_blocks.0.ff.net.0.proj.weight`
- `unet_main.down_blocks.1.motion_modules.0.transformer_blocks.0.ff.net.0.proj.bias`
- `unet_main.down_blocks.1.motion_modules.0.transformer_blocks.0.ff.net.2.weight`
- `unet_main.down_blocks.1.motion_modules.0.transformer_blocks.0.ff.net.2.bias`
- `unet_main.down_blocks.1.motion_modules.0.proj_out.weight`
- ... 487 more

## Uncertain Preserved From SFT Stage2 Sample

- none

## Shape Mismatch

- none

## Missing

- none

## Skipped

- `unet_main.down_blocks.3.attentions: attribute missing on source or destination`
- `unet_main.up_blocks.0.attentions: attribute missing on source or destination`
