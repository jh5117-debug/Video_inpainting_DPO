# Exp42 MiniMax Successful-Removal Visual Review

Status: `MINIMAX_SUCCESSFUL_REMOVAL_POOL_WEAK`

Codex opened the compact temporal evidence sheets generated from the 128-source mining run: 26 success pages and 40 failure pages. Each selected candidate appears in a 16-frame strip with condition, mask/overlay, winner/reference context, and raw MiniMax output columns. This review did not use VOR-Eval, did not use hard comp, and did not launch training.

## Reviewed Evidence

- Success rows reviewed by strip: `52`
- Failure rows reviewed by strip: `80`
- Success scene groups: `18`
- Failure scene groups: `29`
- Success/failure scene overlap: `7`
- Overlap groups: `BLENDER_FOREST026, BLENDER_GRASS001, BLENDER_MOUNTAIN002, REAL_ENV059_00001, REAL_ENV068_00002, REAL_ENV097_00001, REAL_ENV105_00001`
- Full mp4 playback: `false`
- Promotion to Stage2 training data: `false`
- Local compact-sheet archive:
  `/home/hj/exp42_review_assets_archive/20260629_minimax_successful_removal`
- PAI evidence root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp42_pai_minimax_successful_removal_badnoise/official_success_mining_128_retry1_20260629_101104`

## Human Review Counts

- AUTO_FAILURE_LABEL_NOISY_BORDERLINE: `37`
- MEDIUM_HARD_FAILURE_SIGNAL_VISUALLY_PLAUSIBLE: `43`
- SUCCESSFUL_REMOVAL_SIGNAL_VISUALLY_PLAUSIBLE: `52`

## Decision

The mining run found real MiniMax successful-removal signal: visually plausible removals exist under the official 12-step/6-iteration protocol, and the selected candidates are technical-valid. However, success rows are heavily clustered by seed and source, especially forest/grass/indoor staircase scenes. The selected failure set is also technical-valid and useful diagnostically, but many auto-failures are borderline clean at strip scale and require stricter relabeling before they can serve as DPO losers.

Therefore the row-level automatic gate is informative but not sufficient for Exp42 Milestone B/C. The current pool is marked `MINIMAX_SUCCESSFUL_REMOVAL_POOL_WEAK`: bad-noise v3 and Stage2 train/search/shadow data are not unlocked because only `7` overlapping success/failure scene groups were found versus the required `24` usable paired sources.

## Next Minimal Action

Run targeted second-pass mining on source groups that already produced either successes or medium-hard failures, with a pre-registered additional seed budget, to increase same-source success/failure pairs. Do not start SFT/DPO from this pool as-is.
