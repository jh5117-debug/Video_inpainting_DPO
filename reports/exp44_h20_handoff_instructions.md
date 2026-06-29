# Exp44 H20 Handoff Instructions

- Source branch: `research/exp44-pai-minimax-targeted-same-source-mining-20260629`
- Pull/checkout this branch in an isolated H20 worktree.
- Pull only Exp44 manifests, reports, and helper metadata listed below.
- Do not modify PAI Exp44 outputs from H20.
- Do not start GT-only SFT first.
- First H20 experiment: pseudo-success SFT 30-step using the
  pseudo-success distillation train/search/shadow manifests.
- Precision recommendation: fp32 loss reduction; use fp16/bf16 only
  after a one-batch preflight verifies finite latents, finite flow
  target `epsilon - z0`, finite loss, and no NaN/Inf.
- Raw output remains primary; no hard comp may be used to claim gain.
- This handoff is partial and marked `TRAINING_NOT_UNLOCKED` because
  train/search/shadow counts are below 32/16/16.
- This Codex session did not have `/mnt/nas` mounted, so H20 must
  verify that every absolute condition/winner/mask/output path exists
  before launching any dataloader or runner.

## Manifest Checksums

- `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_stage2_gt_distill_train.jsonl`: `24a302b62a72478f4db691559047a6f481a47ce626144c27fd1d51ecb0509e12` (24 rows)
- `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_stage2_gt_distill_search.jsonl`: `4d47e4b3a4177188d24d6a04c117e5b84800a8b8565474e18181f200599dcbc0` (8 rows)
- `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_stage2_gt_distill_shadow.jsonl`: `9aaabdaccb02225edfa138a685a1556771b54150f51512566f0e6298c145bdf0` (8 rows)
- `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_stage2_pseudo_success_train.jsonl`: `e4a9d8dd7b039ee5c3296f6b78811a70202207851ec8a384a9c5fbc9bba69a21` (24 rows)
- `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_stage2_pseudo_success_search.jsonl`: `7fb0ab58b13928379e3d7ecd661b457a57395fa4df04b870fea7d405378715d7` (8 rows)
- `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_stage2_pseudo_success_shadow.jsonl`: `2b6371d028fcd960148208b95357ca644f1fe5c2a6b7d0021fe97f15079c0662` (8 rows)
- `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_stage2_same_source_preference_train.jsonl`: `869bb2ab747ca013c2327957ea26736160d010f3534e1e4be9d3952d0551e374` (24 rows)
- `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_stage2_same_source_preference_search.jsonl`: `3e155c098832100a678afd33772e4e956cb663648561eb2bc6f864373c99d155` (8 rows)
- `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_stage2_same_source_preference_shadow.jsonl`: `68f8543cbd67ee0fc043a7f7aa5d14a1bf0c7694251381e7c98be499bf2ef990` (8 rows)

## H20 Must Also Pull

- `reports/exp44_stage2_dataset_handoff.md`
- `reports/exp44_stage2_dataset_handoff.csv`
- `reports/exp44_stage2_dataset_handoff_summary.json`
- `reports/exp44_badnoise_v4_summary.json`
- `reports/exp44_badnoise_v4_states.csv`
- `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_badnoise_v4_states.jsonl`
