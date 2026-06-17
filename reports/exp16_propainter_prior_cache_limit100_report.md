# Exp16 ProPainter Prior Cache Report

input_manifest: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/exp09_10_11_youtubevos_gtwin_d3comp_pai/manifests/selected_primary_comp.gtwin.pai_paths.jsonl`
output_manifest: `/mnt/nas/hj/H20_Video_inpainting_DPO/data/cache/exp16_propainter_prior_cache_limit100/manifests/exp16_train_with_prior_limit100.jsonl`
rows_written: 100
reused_existing_prior: 0
generated_prior: 100
failed: 0
dry_run: False

This cache stores real ProPainter prior frames only. It must not be
replaced by generated losers or frozen-reference epsilon proxies.
