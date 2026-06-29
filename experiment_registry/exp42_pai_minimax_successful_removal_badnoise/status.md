# Exp42 Status

Current status: `MINIMAX_SUCCESSFUL_REMOVAL_POOL_WEAK`

## 2026-06-29 Readback

- Branch: `research/exp42-pai-minimax-successful-removal-badnoise-20260629`.
- Base: `origin/research/exp40-minimax-psnr-safe-rescue-20260628`.
- Start HEAD: `7dd81ef8baf1377009a4e74b022b9904e2a84957`.
- MiniMax previous state: plumbing-positive, inference-sensitive, protocol
  audited, but no heldout quality-positive recipe.
- Exp40 state read: `MINIMAX_SFT_PSNRSAFE_NEGATIVE`.
- PAI GPU0/GPU1 readback: both free, no compute PID, no cleanup needed.
- MiniMax official repo and weights: present on PAI/NAS.
- Exp41 H20 artifacts were not present in this branch and protected H20
  worktrees were not touched.
- No GPU inference, training, DPO, long run, VOR-Eval use, H20 action, or
  output overwrite was launched by readback.

Report:

- `reports/exp42_pai_minimax_data_readback.md`

Next status target: `MINIMAX_SUCCESSFUL_REMOVAL_POOL_READY`.

## 2026-06-29 Official MiniMax Successful-Removal Mining

- Run root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp42_pai_minimax_successful_removal_badnoise/official_success_mining_128_retry1_20260629_101104`.
- PAI hostname: `dsw-753014-85f54df947-bkp7h`.
- GPU use: GPU0 only; GPU1 available; GPU2-GPU7 untouched; no PID killed.
- Protocol: official MiniMax raw output, `UniPCMultistepScheduler`, fp16,
  `num_inference_steps=12`, `iterations=6`, no CFG, no hard comp, no
  VOR-Eval.
- Sources/candidates: `117` sources, `4` seeds/source, `468` candidates.
- Technical-valid: `468/468`.
- Automatic row-level selected candidates: `52` success, `80` medium-hard
  failure.
- Codex compact-strip review: `26` success pages and `40` failure pages
  opened.
- Source-level review: `18` success scene groups, `29` failure scene groups,
  `7` success/failure overlap groups.
- Failure label caveat: `37/80` auto-failure rows looked visually borderline
  or metric/boundary driven at strip scale.

Decision:

- `MINIMAX_SUCCESSFUL_REMOVAL_POOL_WEAK`
- Bad-noise v3: not unlocked.
- Stage2 train/search/shadow: not unlocked.
- Short SFT/DPO: not unlocked.

Reports:

- `reports/exp42_minimax_official_successful_removal_mining.md`
- `reports/exp42_minimax_successful_removal_visual_review.md`
- `reports/exp42_minimax_successful_removal_visual_review.csv`
- `reports/exp42_minimax_successful_removal_summary.json`

Next minimal action: targeted second-pass same-source success/failure mining
with stricter per-video relabeling.
