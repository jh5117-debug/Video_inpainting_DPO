# Status

- status: `h20_stage2_running_step580_posthoc_davis_watcher_required`
- conclusion: Planned correction for Exp7 before expanding D3; current run tests Stage1+Stage2 with the same regularized full-loss setting as Exp8. The original H20 launch script ran two-stage training only and skipped the intended Exp8-style DAVIS validation handoff, so a posthoc DAVIS watcher must be launched from synced git code.
- next_action: Let Stage2 finish, then run canonical `experiment_registry/exp07_fix_smallmask_prior/code/posthoc_davis_val_h20.sh` to validate `DPO-S1_SFT-S2` and `DPO-S1_DPO-S2` on DAVIS without using GPU 0. The old `scripts/run_exp07_fix_smallmask_prior_posthoc_davis_val_h20.sh` path is a compatibility wrapper only.

## H20 data generation

- status: running
- pid: `2590851`
- log: `/home/nvme01/H20_Video_inpainting_DPO/logs/pipelines/exp07_fix_smallmask_prior_data_generation_h20_20260605_050336.log`
- output_root: `/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/exp07_fix_videodpo_smallmask15_20_prior_k4`
- selected_primary_comp manifest is available on H20 with 1000 rows.
- training: H20 Stage1+Stage2 launched 2026-06-06 14:25 CST.
- run_version: `20260606_142555`
- code_commit: `898f9c8`
- clean_worktree: `/home/nvme01/H20_Video_inpainting_DPO_exp8c_pai_sync`
- pipeline_log: `/home/nvme01/H20_Video_inpainting_DPO/logs/pipelines/exp07_fix_smallmask_prior_wingap_lose025_s1s2_2000_h20_20260606_142555.log`
- pid_file: `/home/nvme01/H20_Video_inpainting_DPO/logs/pipelines/exp07_fix_smallmask_prior_wingap_lose025_s1s2_2000_h20.pid`
- stage1_log: `/home/nvme01/H20_Video_inpainting_DPO/logs/train/exp07_fix_smallmask_prior_wingap_lose025_s1s2_2000_h20/stage1.log`
- stage1_run_dir: `/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260606_142555_exp07_fix_smallmask_prior_wingap_lose025_s1s2_2000_h20_stage1`
- monitored_status: Stage1 completed `global_step=2000`; Stage2 is running and reached `global_step=580` on 2026-06-07 02:38 CST. `dpo_diagnostics.csv` is present for both stages, GPUs 1-7 are used, GPU 0 remains reserved, and no SIGFPE/OOM/Traceback was observed.
- cleanup_report: `/home/nvme01/H20_Video_inpainting_DPO/reports/h20_cleanup_before_exp7_20260606_142113.md`
- stage2_log: `/home/nvme01/H20_Video_inpainting_DPO/logs/train/exp07_fix_smallmask_prior_wingap_lose025_s1s2_2000_h20/stage2.log`
- stage2_run_dir: `/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260606_142555_exp07_fix_smallmask_prior_wingap_lose025_s1s2_2000_h20_stage2`
- stage1_davis_val_status: missing from original launcher; must be recovered posthoc as `DPO-S1_SFT-S2`.
- stage2_davis_val_status: pending Stage2 completion; must be recovered posthoc as `DPO-S1_DPO-S2`.

## H20 precision and GPU rule

- GPU 0 is reserved; use `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7`.
- Use the SIGFPE-safe profile from PRD:
  `MIXED_PRECISION=no`, `POLICY_DTYPE=fp32`, `VAE_DTYPE=fp32`,
  `REF_DTYPE=fp32`, `TEXT_DTYPE=fp32`, `SPLIT_POS_NEG_FORWARD=0`.
