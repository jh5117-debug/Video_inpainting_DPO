# Exp37 LocalDPO-style OR Corruption Pool

Status: `LOCALDPO_STYLE_POOL_READY_VISUAL_REVIEW_PASS`

Built deterministic local corruptions from VOR-Train style Gate64 rows only:

- condition = `V_obj`
- winner = `V_bg`
- loser = locally corrupted `V_bg`
- mask = object mask
- affected map = `abs(V_obj - V_bg)` for profile construction
- candidate rows per source <= 2
- VOR-Eval used = `false`
- far outside is visually preserved by exact winner reinjection outside the local soft region

## Automatic vs Codex Final Classification

- Automatic selected usable: `39/48`.
- Codex final usable after visual review: `48/48`.
- Reason for difference: automatic outside PSNR counted affected/boundary soft-region changes as outside damage on 9 rows; visual review found localized hard-but-plausible defects without global collapse or far-outside damage.

## Final Split Summary

- Train32 final classification counts: `{'HARD_BUT_PLAUSIBLE': 8, 'MEDIUM_HARD_ELIGIBLE': 24}`
- Heldout16 final classification counts: `{'MEDIUM_HARD_ELIGIBLE': 14, 'HARD_BUT_PLAUSIBLE': 2}`
- Train32 SHA256: `1ed5c7d4667e7ad1ddc26a042a8613a2a3135c8a8bd3da37071e39e608b66269`
- Heldout16 SHA256: `c761ec3115bf28305879cfd5a9ea835eb121cb9f95ece1e1dcdce99e95ff4abf`

## Visual Review

Codex opened all 48 selected primary review sheets in six batches. The selected corruptions are localized to object, affected, or boundary/effect regions; no black/purple collapse, global frame damage, or systematic far-outside breakage was observed. Some rows are intentionally hard, especially REAL human/animal cases with visible residual silhouettes, and are labeled `HARD_BUT_PLAUSIBLE` rather than medium-hard.

Artifacts:

- Candidate metrics: `reports/exp37_localdpo_style_or_corruption_pool.csv`
- Visual review: `reports/exp37_localdpo_style_visual_review.csv`
- Summary: `reports/exp37_localdpo_style_or_corruption_pool_summary.json`
- Train manifest: `exp37_minimax_localdpo_badnoise_rescue/manifests/localdpo_or_train32.jsonl`
- Heldout manifest: `exp37_minimax_localdpo_badnoise_rescue/manifests/localdpo_or_heldout16.jsonl`
