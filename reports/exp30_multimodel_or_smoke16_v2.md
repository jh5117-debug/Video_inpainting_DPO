# Exp30 Multi-Model OR Smoke16 V2

Status: `MULTIMODEL_OR_SMOKE16_V2_BLOCKED`

Reason: controlled corruption usable fallback count is 5/16, below preregistered >=6/16 requirement; Gate64 and adapter gates remain stopped

## Gate Checks

- Non-EffectErase technical-valid candidates: 32/32.
- Non-EffectErase usable candidates: 9/32.
- Controlled corruption usable fallback: 5/16, required >= 6: **FAIL**.
- MiniMax usable: 4/16, required >= 3 or documented low-yield: pass.
- Systemic decode/frame/mask alignment failure: no.

Because the controlled-corruption fallback criterion failed, `Gate64` is not
unlocked. ProPainter and DiffuEraser were not used to override this gate;
DiffuEraser OR remains pending verified stack status, and EffectErase remains
diagnostic-only.

## Model Counts

- Controlled corruption: `{'candidate_count': 16, 'technical_valid': 16, 'usable': 5, 'classification_counts': {'TRIVIAL_BAD': 11, 'HARD_BUT_PLAUSIBLE': 2, 'MEDIUM_HARD_ELIGIBLE': 3}}`
- MiniMax official: `{'candidate_count': 16, 'technical_valid': 16, 'usable': 4, 'classification_counts': {'TRIVIAL_BAD': 12, 'MEDIUM_HARD_ELIGIBLE': 3, 'HARD_BUT_PLAUSIBLE': 1}}`

## Codex Visual Review

Codex opened 32/32 per-sample temporal strips locally: 16 controlled-corruption
and 16 MiniMax. Controlled corruption preserved outside pixels by construction
but often produced hard local residuals or temporal artifacts. MiniMax produced
some usable local-defect examples, but many outputs were too close, had residual
objects, or showed black/smudged local artifacts.

## Outputs

- `reports/exp30_multimodel_or_smoke16_v2.csv`
- `reports/exp30_multimodel_or_smoke16_metrics_v2.csv`
- `reports/exp30_multimodel_or_smoke16_visual_review_v2.csv`
- `reports/exp30_multimodel_or_smoke16_summary_v2.json`
- Controlled assets: `reports/exp30_controlled_corruption_smoke16_v2_assets/`
- MiniMax assets: `reports/exp30_minimax_smoke16_v2_assets/`
