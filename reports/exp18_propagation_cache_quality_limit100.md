# Exp18 Multi-frame Propagation Cache Report

- input_manifest: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/exp09_10_11_youtubevos_gtwin_d3comp_pai/manifests/selected_primary_comp.gtwin.pai_paths.jsonl`
- output_manifest: `/mnt/nas/hj/H20_Video_inpainting_DPO/data/cache/exp18_multiframe_propagation_cache_limit100/manifests/exp18_train_with_multiframe_prop_limit100.jsonl`
- output_root: `/mnt/nas/hj/H20_Video_inpainting_DPO/data/cache/exp18_multiframe_propagation_cache_limit100`
- limit: `100`
- nframes: `16`
- method: `farneback_multisource_agreement`
- source_window: `3`
- tau_conf: `0.5`
- write_oracle: `True`

## Summary

- total attempted: `100`
- ok/reused: `100`
- failed: `0`
- mean propagation coverage: `0.038680`
- mean confidence: `0.037560`
- mean avg_num_sources_used: `0.228855`
- mean propagated_region_psnr: `23.674436`
- mean full_mask_prop_psnr: `30.754745`

## Interpretation Rule

If mean propagation coverage is below 0.05 or propagated-region PSNR is low,
the propagation cache should be treated as not useful and Exp18 training should not start.

## Post-run Note

The PAI gate did continue into the preplanned Stage1-500 diagnostic variants so
that Exp18a/b/c could be compared under the same DAVIS10 sanity protocol. The
low non-oracle coverage was confirmed by dpo-diag and is part of the final
negative-ablation conclusion in `reports/exp18_final_pai_gate_report.md`.
