# Exp18 PAI Precheck Report

- timestamp: 2026-06-18T05:06:27+08:00
- host: dsw-753014-dc85766cb-4v2jj
- repo: /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp18_gate
- sync_strategy: clean_worktree_plus_hal_git_archive
- source_commit: 1ff7246 Add Exp18 multi-frame propagation gated DPO
- syntax: python py_compile and bash -n passed
- train_manifest: /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/exp09_10_11_youtubevos_gtwin_d3comp_pai/manifests/selected_primary_comp.gtwin.pai_paths.jsonl
- davis_eval: /mnt/workspace/hj/nas_hj/data/external/davis_432_240
- sft48000: /mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000
- exp11_checkpoint: /mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s2_2000_davis_pai/last_weights
- selected_gpu: 0
- gpu_status: GPU 0 idle at precheck
- precheck_result: PASS
