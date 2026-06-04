# Commands

No command is run from this registry file. Actual training/eval commands must be copied into this file before launch.

## Current command status

Generate/check smallmask data, then H20 Stage1 gate if data is ready.

## H20 data generation command

Launched 2026-06-05 CST with `tools/videodpo_generated_loser_calibration.py --models diffueraser --limit 1000 --mask_policy_config configs/generation/videodpo_partialmask_policy_v2_smallmask15_20_k4.yaml`. See report `reports/h20_exp07_fix_smallmask_prior_status.md`.
