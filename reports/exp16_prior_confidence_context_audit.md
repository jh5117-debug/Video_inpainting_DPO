# Exp16 Prior-Confidence Context Audit

Date: 2026-06-17

## 1. Current Best Exp11 Outer B0.75 S2

Current best:

```text
Exp11 boundary outer b0.75 S2
```

Code:

```text
exp11_region_boundary_ablation/code/train_stage1.py
exp11_region_boundary_ablation/code/train_stage2.py
```

Checkpoint / run paths from registry:

```text
Stage1: /mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s1_2000_davis_pai
Stage2: /mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s2_2000_davis_pai
Stage2 last_weights: /mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s2_2000_davis_pai/last_weights
```

Metrics:

```text
DAVIS50 PSNR 33.013954, SSIM 0.972295, LPIPS 0.015363, VFID 0.175423, TC 0.971122
YouTubeVOS100 PSNR 33.7238, SSIM 0.9711, LPIPS 0.0168, VFID 0.1925, TC 0.9821
```

## 2. Current Region-Local Loss Implementation

Implementation location:

```text
exp11_region_boundary_ablation/code/train_stage1.py
  compute_dpo_loss()
  build_region_loss_weight_map()

exp11_region_boundary_ablation/code/train_stage2.py
  compute_dpo_loss()
  build_region_loss_weight_map()
```

The current loss computes region-weighted MSE between predicted diffusion noise
and the sampled noise target:

```text
m = sum(region_weight_map * mse_map) / (sum(region_weight_map) + eps)
```

It then applies log-ratio normalized DPO:

```text
g_w = log((m_w + eps)/(m_w_ref + eps))
g_l = log((m_l + eps)/(m_l_ref + eps))
```

## 3. Boundary Outer B0.75 Definition

Mask convention in the current DiffuEraser DPO dataset:

```text
brushnet mask: 0 = hole / unknown, 1 = known outside
hole = 1 - brushnet_mask
```

Exp11 outer boundary:

```text
boundary_outer = dilate(hole) - hole
mask_weight = 1.0
boundary_weight = 0.75
outside_weight = 0.05
```

This is the setting inherited by Exp16.

## 4. Can Current Training Code Get Predicted X0 / Clean Latent?

Current Exp11 code encodes GT / loser latents and uses:

```text
noise_scheduler.add_noise(latents, noise, timesteps)
model_pred = UNet / BrushNet prediction
target = noise
```

It does not currently reconstruct `z_hat_x0` or compute image/latent-space
ProPainter-prior consistency. Exp16 adds an isolated helper:

```text
exp16_prior_confidence_gated_dpo/code/exp16_loss.py
  predict_x0_from_model_output()
```

But the full Stage1 / Stage2 trainer integration is not yet passed.

## 5. Does The Current Manifest Already Have ProPainter Prior Paths?

The known PAI training manifest is:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4/manifests/selected_primary_comp.repaired.pai_paths.jsonl
```

Earlier PAI sampling showed rows with:

```text
win_video_path
final_loser_video_path
raw_loser_video_path
comp_loser_video_path
mask_path
diffueraser_prior_mode=unconfirmed
```

No verified `prior_frame_dir` / `propainter_frame_dir` field was found in the
sampled rows. A long PAI path search showed generated-loser candidate folders
with `diffueraser`, `mask`, and `win` directories; no verified ProPainter prior
cache has been accepted for Exp16.

Conclusion: Exp16 must build or locate a real ProPainter prior cache before any
training.

## 6. Can DiffuEraser Pipeline Export ProPainter Prior Frames?

Yes. Existing code path:

```text
tools/run_davis50_framewise_protocol_eval.py
inference/run_BR.py
DPO_finetune/infer_propainter_candidate.py
```

`tools/run_davis50_framewise_protocol_eval.py` calls ProPainter and keeps
`prior_frames` in memory before DiffuEraser inference. For Exp16 training, a
cache script has been added:

```text
exp16_prior_confidence_gated_dpo/code/precompute_propainter_prior_cache.py
```

It saves real ProPainter prior frames and writes a new manifest with
`prior_frame_dir`.

## 7. DPO Diagnostics Extensibility

The existing dpo_diag writer already accepts dictionaries of scalar diagnostics.
Exp16 can add:

```text
L_prior
L_gen
L_boundary_extra
prior_conf_mean/p10/p50/p90
reliable_area_ratio
generate_area_ratio
prior_target_mode
confidence_mode
```

## 8. Evaluation Reuse

The fixed DAVIS / YouTubeVOS inpainting protocol can be reused:

```text
tools/run_davis50_framewise_protocol_eval.py
scripts/run_exp11_outer_youtubevos100_framewise_protocol_pai.sh
inference/metrics.py
```

No VBench should be used for Exp16.

## Audit Conclusion

Exp16 is a plausible next direction, but current evidence is not sufficient to
launch training yet. The hard blockers are:

1. no verified real ProPainter prior manifest for the training rows;
2. the copied Exp11 training scripts still compute epsilon/noise DPO, so full
   x0 prior loss integration must pass before Stage1/Stage2 launch.

