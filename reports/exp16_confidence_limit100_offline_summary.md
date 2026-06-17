# Exp16 Prior Confidence Limit100 Audit

manifest: `/mnt/nas/hj/H20_Video_inpainting_DPO/data/cache/exp16_propainter_prior_cache_limit100/manifests/exp16_train_with_prior_limit100.jsonl`
rows_ok: 100
rows_failed: 0
confidence_mode: `gt_error`
confidence_alpha: 5.0

| metric | mean |
|---|---:|
| prior_conf_mean | 0.656014 |
| prior_conf_p10 | 0.239536 |
| prior_conf_p50 | 0.725268 |
| prior_conf_p90 | 0.940553 |
| prior_conf_mean_inside_mask | 0.656014 |
| prior_conf_std_inside_mask | 0.264408 |
| prior_conf_p10_inside_mask | 0.239536 |
| prior_conf_p50_inside_mask | 0.725268 |
| prior_conf_p90_inside_mask | 0.940553 |
| reliable_area_ratio | 0.256022 |
| generate_area_ratio | 0.254534 |
| reliable_weight_mass | 0.656014 |
| generate_weight_mass | 0.343986 |
| reliable_generate_mass_sum | 1.000000 |
| mask_area_ratio | 0.256022 |
| boundary_area_ratio | 0.008180 |
