# Exp30 Controlled Corruption V3 Calibration Plan

Status: `CONTROLLED_CORRUPTION_V3_PLAN_LOCKED`

No new video generation, GPU task, Smoke32, Gate64, MiniMax adapter gate, or
training was run for this milestone.

## Motivation

Smoke16 v2 showed that controlled corruption was technically valid but too
often failed the medium-hard target:

- Technical-valid: 16/16.
- Usable: 5/16.
- Dominant blocker: temporal discontinuity in 11/16 rows.
- Outside preservation was not the dominant blocker because outside pixels were
  reinjected from the clean winner.

The v3 goal is to turn controlled corruption into a calibrated fallback loser
source with bounded local defects, not to create a final method or ground truth.

## Locked Profiles

| profile | role | region | noise | condition mix | blur | feather | temporal smoothing | enabled in smoke16_v3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CC-v3-A | mild-object | object mask | 2.5 | 0.25 | 21 | 9 px | EMA alpha 0.75 | yes, repair subset |
| CC-v3-B | medium-object | object mask | 4.0 | 0.35 | 17 | 9 px | EMA alpha 0.65 | yes, all sources |
| CC-v3-C | affected-soft | object mask union soft affected map | 3.0 | 0.25 | 19 | 11 px | EMA alpha 0.75 | yes, affected subset |
| CC-v3-D | boundary-focused | object boundary plus affected edge | 3.5 | 0.40 | 13 | 5 px | EMA alpha 0.70 | no, reserve only |

All profiles use strict outside winner reinjection outside the feathered
corruption region. No profile may use hard comp as the primary loser.

## Smoke16 V3 Controlled Schedule

The source rows remain the repaired smoke16 v2 rows unless a row becomes
technically invalid.

- Run CC-v3-B for all 16 sources.
- Run CC-v3-A for six v2 temporal-discontinuity rows:
  `BLENDER_FOREST016_00001`, `BLENDER_FOREST006_00001`,
  `BLENDER_FOREST008_00001`, `BLENDER_FOREST015_00001`,
  `BLENDER_FOREST018_00001`, `REAL_ENV045_00003_001_01`.
- Run CC-v3-C for two affected-region rows:
  `REAL_ENV045_00003_001_01`, `REAL_ENV046_00002_001_01`.
- Do not run CC-v3-D in smoke16 v3.
- Controlled candidates in smoke16 v3: at most 24.
- Per-source controlled candidates: at most 2.

This deliberately avoids running all profiles on all sources and prevents
unbounded sample search.

## Success Target

For controlled fallback to stop blocking smoke16 v3:

- Technical-valid source coverage: at least 15/16.
- Usable source coverage: at least 8/16.
- Trivial-bad count in the selected one-controlled-per-source view: at most
  6/16.
- Outside systematic damage: 0.

The full smoke16 v3 promotion gate still also requires multi-model usable
coverage and at least two non-EffectErase generator families contributing
usable candidates.

## Claim Boundary

Controlled corruption is a fallback/data-source mechanism for medium-hard loser
construction. It is not a final model, not ground truth, and cannot by itself
unlock MiniMax adapter claims.
