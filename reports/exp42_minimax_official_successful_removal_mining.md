# Exp42 MiniMax Official Successful-Removal Mining

Automatic mining status: `MINIMAX_SUCCESSFUL_REMOVAL_POOL_READY`

Codex post-review decision: `MINIMAX_SUCCESSFUL_REMOVAL_POOL_WEAK`

This milestone ran official MiniMax raw inference only. It did not train,
did not use VOR-Eval, did not use hard comp, and did not modify the
official MiniMax repository or shared metric code.

## Protocol

- Sources: `117`
- Seeds per source: `4`
- Candidates: `468`
- Scheduler: `UniPCMultistepScheduler`
- num_inference_steps: `12`
- iterations: `6`
- dtype: `float16`
- raw output primary: `true`
- VOR-Eval used: `false`

## Automatic Mining Counts

- Successful candidates: `52`
- Medium-hard failure candidates: `80`
- Technical-invalid candidates: `0`

Automatic labels are provisional. Codex opened the compact 16-frame temporal
evidence sheets for all selected success/failure rows after the run. The
review found real MiniMax successful-removal signal, but also heavy source/seed
clustering and noisy auto-failure labels. Source-level success/failure overlap
is only `7` scene groups, below the `>=24` usable-pair gate required for
bad-noise v3 or Stage2 data construction.

Therefore this milestone is informative but not training-ready. Do not start
Stage2 SFT/DPO from these manifests as-is.

## Codex Visual Review Decision

- Success rows reviewed by compact strip: `52`
- Failure rows reviewed by compact strip: `80`
- Success scene groups: `18`
- Failure scene groups: `29`
- Success/failure scene-group overlap: `7`
- Auto-failure rows judged visually noisy/borderline at strip scale: `37`
- Full mp4 playback: `false`
- Stage2 / bad-noise v3 unlocked: `false`

Final Milestone A status: `MINIMAX_SUCCESSFUL_REMOVAL_POOL_WEAK`.

## Outputs

- `exp42_pai_minimax_successful_removal_badnoise/manifests/exp42_minimax_successful_candidates_all.jsonl`
- `exp42_pai_minimax_successful_removal_badnoise/manifests/exp42_minimax_successful_candidates_selected.jsonl`
- `exp42_pai_minimax_successful_removal_badnoise/manifests/exp42_minimax_failure_candidates_selected.jsonl`
- `reports/exp42_minimax_official_successful_removal_mining.csv`
- `reports/exp42_minimax_successful_removal_summary.json`
- `reports/exp42_minimax_successful_removal_visual_review.csv`
- `reports/exp42_minimax_successful_removal_visual_review.md`

NAS evidence root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp42_pai_minimax_successful_removal_badnoise/official_success_mining_128_retry1_20260629_101104`
