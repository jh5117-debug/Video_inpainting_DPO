# Status

- status: `h20_s1s2_launching`
- conclusion: Planned correction for Exp7 before expanding D3; current run should test Stage1+Stage2 with the same regularized full-loss setting as Exp8.
- next_action: Stop H20 Exp8c, use GPUs 1-7, and launch `scripts/launch_exp07_fix_smallmask_prior_wingap_s1s2_2000_h20.sh` from a clean git-synced worktree.

## H20 data generation

- status: running
- pid: `2590851`
- log: `/home/nvme01/H20_Video_inpainting_DPO/logs/pipelines/exp07_fix_smallmask_prior_data_generation_h20_20260605_050336.log`
- output_root: `/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/exp07_fix_videodpo_smallmask15_20_prior_k4`
- selected_primary_comp manifest is available on H20 with 1000 rows.
- training: H20 Stage1+Stage2 restart pending launch.

## H20 precision and GPU rule

- GPU 0 is reserved; use `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7`.
- Use the SIGFPE-safe profile from PRD:
  `MIXED_PRECISION=no`, `POLICY_DTYPE=fp32`, `VAE_DTYPE=fp32`,
  `REF_DTYPE=fp32`, `TEXT_DTYPE=fp32`, `SPLIT_POS_NEG_FORWARD=0`.
