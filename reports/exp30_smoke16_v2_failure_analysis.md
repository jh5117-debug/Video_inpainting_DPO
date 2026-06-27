# Exp30 Smoke16 V2 Failure Analysis

Status: `SMOKE16_V2_FAILURE_ANALYZED`

No new generation, GPU inference, Gate64, adapter gate, or training was run for this milestone.

## Overall

- Candidates analyzed: 32.
- Technical-valid candidates: 32.
- Usable candidates: 9.
- Classification counts: `{'TRIVIAL_BAD': 23, 'HARD_BUT_PLAUSIBLE': 3, 'MEDIUM_HARD_ELIGIBLE': 6}`.
- Smoke16 v2 remains blocked because controlled corruption produced only 5/16 usable fallback candidates, below the preregistered >=6/16 threshold.
- Continuation v3 visual readback opened 4 controlled-corruption overview pages and 4 MiniMax review pages, covering all 32 v2 candidates.

## Controlled Corruption

- Technical-valid: 16/16.
- Usable: 5/16.
- Failure counts: `{'temporal discontinuity': 11, 'local residual too sharp': 2, 'bounded local texture mismatch': 3}`.
- Main failure: excessive temporal discontinuity from a single aggressive frame-wise corruption profile.
- Outside preservation was not the issue for controlled corruption; outside pixels were effectively reinjected.

## MiniMax Official

- Technical-valid: 16/16.
- Usable: 4/16.
- Failure counts: `{'outside damage': 7, 'bounded residual / medium-hard': 3, 'temporal flicker / instability': 4, 'too bad': 1, 'strong but plausible local defect': 1}`.
- Main failures: outside damage, temporal flicker/instability, too-bad local outputs, residual object/effect, and occasional smudged/black artifacts.
- MiniMax should remain part of a multi-model candidate pool; it is not sufficient as the only loser generator.

## Samples Suitable For V3 Planning

- `controlled_corruption` usable references: BLENDER_FOREST007_00001, REAL_ENV044_00005_001_01, REAL_ENV046_00003_001_01, REAL_ENV046_00004_001_01, REAL_ENV046_00005_001_01
- `minimax_official` usable references: BLENDER_FOREST007_00001, REAL_ENV045_00001_001_01, REAL_ENV045_00002_001_01, REAL_ENV046_00003_001_01

## Samples To Retain But Not Tune Against

The same repaired smoke16 source rows should remain locked for smoke16 v3, but the v2 per-candidate outputs must not be used to cherry-pick source replacements. The v2 failures guide preregistered generator changes only.

## Preregistered Fix Items For Smoke16 V3

- controlled corruption must reduce frame-independent noise and add temporal smoothing.
- controlled corruption must include mild/medium object and affected-soft profiles, not one aggressive profile.
- MiniMax should remain one family in a multi-model pool, not the sole source of training losers.
- DiffuEraser and ProPainter may be enabled only after verified stack smoke.
- EffectErase remains diagnostic-only and excluded from smoke promotion/training pairs.

## Per-Candidate Failure Table

| model | sample_id | classification | failure_category | v3_recommendation |
| --- | --- | --- | --- | --- |
| controlled_corruption | BLENDER_FOREST006_00001 | TRIVIAL_BAD | temporal discontinuity | Reduce noise/condition mix and add temporal smoothing. |
| controlled_corruption | BLENDER_FOREST007_00001 | HARD_BUT_PLAUSIBLE | local residual too sharp | Keep as hard-plausible reference; v3 should soften this profile. |
| controlled_corruption | BLENDER_FOREST008_00001 | TRIVIAL_BAD | temporal discontinuity | Reduce noise/condition mix and add temporal smoothing. |
| controlled_corruption | BLENDER_FOREST015_00001 | TRIVIAL_BAD | temporal discontinuity | Reduce noise/condition mix and add temporal smoothing. |
| controlled_corruption | BLENDER_FOREST016_00001 | TRIVIAL_BAD | temporal discontinuity | Reduce noise/condition mix and add temporal smoothing. |
| controlled_corruption | BLENDER_FOREST017_00001 | TRIVIAL_BAD | temporal discontinuity | Reduce noise/condition mix and add temporal smoothing. |
| controlled_corruption | BLENDER_FOREST018_00001 | TRIVIAL_BAD | temporal discontinuity | Reduce noise/condition mix and add temporal smoothing. |
| controlled_corruption | BLENDER_FOREST019_00001 | TRIVIAL_BAD | temporal discontinuity | Reduce noise/condition mix and add temporal smoothing. |
| controlled_corruption | REAL_ENV044_00005_001_01 | HARD_BUT_PLAUSIBLE | local residual too sharp | Keep as hard-plausible reference; v3 should soften this profile. |
| controlled_corruption | REAL_ENV045_00001_001_01 | TRIVIAL_BAD | temporal discontinuity | Reduce noise/condition mix and add temporal smoothing. |
| controlled_corruption | REAL_ENV045_00002_001_01 | TRIVIAL_BAD | temporal discontinuity | Reduce noise/condition mix and add temporal smoothing. |
| controlled_corruption | REAL_ENV045_00003_001_01 | TRIVIAL_BAD | temporal discontinuity | Reduce noise/condition mix and add temporal smoothing. |
| controlled_corruption | REAL_ENV046_00002_001_01 | TRIVIAL_BAD | temporal discontinuity | Reduce noise/condition mix and add temporal smoothing. |
| controlled_corruption | REAL_ENV046_00003_001_01 | MEDIUM_HARD_ELIGIBLE | bounded local texture mismatch | Keep as positive reference for v3. |
| controlled_corruption | REAL_ENV046_00004_001_01 | MEDIUM_HARD_ELIGIBLE | bounded local texture mismatch | Keep as positive reference for v3. |
| controlled_corruption | REAL_ENV046_00005_001_01 | MEDIUM_HARD_ELIGIBLE | bounded local texture mismatch | Keep as positive reference for v3. |
| minimax_official | BLENDER_FOREST006_00001 | TRIVIAL_BAD | outside damage | Do not use raw candidate unless outside preservation improves. |
| minimax_official | BLENDER_FOREST007_00001 | MEDIUM_HARD_ELIGIBLE | bounded residual / medium-hard | Keep as usable MiniMax candidate. |
| minimax_official | BLENDER_FOREST008_00001 | TRIVIAL_BAD | outside damage | Do not use raw candidate unless outside preservation improves. |
| minimax_official | BLENDER_FOREST015_00001 | TRIVIAL_BAD | temporal flicker / instability | Try fewer iterations or smoother mask/seed only under preregistration. |
| minimax_official | BLENDER_FOREST016_00001 | TRIVIAL_BAD | outside damage | Do not use raw candidate unless outside preservation improves. |
| minimax_official | BLENDER_FOREST017_00001 | TRIVIAL_BAD | temporal flicker / instability | Try fewer iterations or smoother mask/seed only under preregistration. |
| minimax_official | BLENDER_FOREST018_00001 | TRIVIAL_BAD | temporal flicker / instability | Try fewer iterations or smoother mask/seed only under preregistration. |
| minimax_official | BLENDER_FOREST019_00001 | TRIVIAL_BAD | temporal flicker / instability | Try fewer iterations or smoother mask/seed only under preregistration. |
| minimax_official | REAL_ENV044_00005_001_01 | TRIVIAL_BAD | too bad | Reject; local result too far from clean winner. |
| minimax_official | REAL_ENV045_00001_001_01 | MEDIUM_HARD_ELIGIBLE | bounded residual / medium-hard | Keep as usable MiniMax candidate. |
| minimax_official | REAL_ENV045_00002_001_01 | HARD_BUT_PLAUSIBLE | strong but plausible local defect | Keep as usable MiniMax candidate. |
| minimax_official | REAL_ENV045_00003_001_01 | TRIVIAL_BAD | outside damage | Do not use raw candidate unless outside preservation improves. |
| minimax_official | REAL_ENV046_00002_001_01 | TRIVIAL_BAD | outside damage | Do not use raw candidate unless outside preservation improves. |
| minimax_official | REAL_ENV046_00003_001_01 | MEDIUM_HARD_ELIGIBLE | bounded residual / medium-hard | Keep as usable MiniMax candidate. |
| minimax_official | REAL_ENV046_00004_001_01 | TRIVIAL_BAD | outside damage | Do not use raw candidate unless outside preservation improves. |
| minimax_official | REAL_ENV046_00005_001_01 | TRIVIAL_BAD | outside damage | Do not use raw candidate unless outside preservation improves. |
