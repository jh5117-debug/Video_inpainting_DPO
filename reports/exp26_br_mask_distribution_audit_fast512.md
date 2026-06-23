# Exp26 BR Mask Distribution Audit

- historical manifest: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4/manifests/selected_primary_comp.repaired.pai_paths.jsonl`
- probe manifest: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/49f_probe_7f9ec40/vp2_probe4_49f_masks.jsonl`
- historical sample limit: 512

## Summary
### historical_br_youtubevos_k4_limit512
- ok=512 failed=0
- area_mean: mean=0.254348 p10=0.212689 p50=0.253986 p90=0.291199
- edge_touch_ratio: mean=0.222656 p10=0.000000 p50=0.000000 p90=1.000000
- centroid_step_motion_mean: mean=0.000941 p10=0.000000 p50=0.000000 p90=0.002815
- first_frame_area: mean=0.254331 p10=0.212689 p50=0.253986 p90=0.291113
- bbox_w_mean: mean=0.504323 p10=0.431396 p50=0.498047 p90=0.585205
- bbox_h_mean: mean=0.786556 p10=0.668750 p50=0.790625 p90=0.900000

### exp26_probe4_generated_ellipse
- ok=4 failed=0
- area_mean: mean=0.136569 p10=0.066915 p50=0.159399 p90=0.206634
- edge_touch_ratio: mean=0.051020 p10=0.000000 p50=0.000000 p90=0.204082
- centroid_step_motion_mean: mean=0.005325 p10=0.004891 p50=0.004996 p90=0.006494
- first_frame_area: mean=0.000000 p10=0.000000 p50=0.000000 p90=0.000000
- bbox_w_mean: mean=0.297070 p10=0.228651 p50=0.305006 p90=0.361958
- bbox_h_mean: mean=0.556916 p10=0.365136 p50=0.653061 p90=0.726190

## Decision
Probe4 masks are ellipse-only and valid for plumbing, but Gate16/Gate64 should use a mixed BR mask protocol calibrated from historical area, edge-touch, and motion buckets.
