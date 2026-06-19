# Exp20 Legacy 10-Step Smoke

- status: REAL_10STEP_SMOKE_PASSED
- checkpoint_reload: CHECKPOINT_RELOAD_PASSED
- run_dir: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp20_autoresearch_scale_adaptive_region_dpo/smoke/legacy_exact_10step_smoke_20260620_000214`
- log: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp20/legacy_exact_10step_smoke_20260620_000214.log`
- gpu: `0`
- manifest: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/exp09_10_11_youtubevos_gtwin_d3comp_pai/manifests/selected_primary_comp.gtwin.pai_paths.jsonl`
- effective_global_batch: 4
- world_size: 1
- per_device_batch: 1
- gradient_accumulation_steps: 4
- max_train_steps: 10
- checkpoint: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp20_autoresearch_scale_adaptive_region_dpo/smoke/legacy_exact_10step_smoke_20260620_000214/checkpoint-10`
- last_weights: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp20_autoresearch_scale_adaptive_region_dpo/smoke/legacy_exact_10step_smoke_20260620_000214/last_weights`
- dpo_diag: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp20_autoresearch_scale_adaptive_region_dpo/smoke/legacy_exact_10step_smoke_20260620_000214/dpo_diagnostics.csv`

The run used real DiffuEraser Stage1 models, SFT-48000 initialization/reference, shared winner/loser noise and timesteps, frozen reference, and Exp20 legacy_exact region maps.
