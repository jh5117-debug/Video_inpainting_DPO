# Exp27 Candidate Methods V2

## Primary Candidate: Restoration-Critical Failure-Structured Preference Optimization

Short name: `RC-FPO`

Core idea:

Build preference pairs from actual inpainting/object-removal failure modes and optimize them with a restoration-critical region decomposition. The novelty is not "localized DPO"; it is the pairing of task-native failure mining with BR/OR-specific region semantics and anti-shortcut diagnostics.

Data:

- BR: condition is masked GT video; winner is clean GT; loser is raw inpainting output or faithful LocalDPO-style local corruption baseline.
- OR: condition is `V_obj`; winner is `V_bg`; mask is object mask; loser is raw removal output. No hard comp.
- Each loser receives a defect vector for object residual, affected-region error, seam, blur, copy-condition, copy-winner, outside damage, flicker, TC, and Ewarp.

Objective:

- Use Current LoVI as the initial objective.
- Add region decomposition only after exact LocalDPO/SDPO/Linear-DPO baselines are available.
- Region definitions:
  - BR: mask core, outer boundary/seam, outside context.
  - OR: object mask, affected region, unaffected background.

Novelty boundary:

- Different from LocalDPO because regions and losers are task-native, not random local corruption.
- Different from SDPO because safety is not the core claim.
- Different from Linear-DPO because utility/reference update is not the core claim.

First experiments:

- D0 task-native self loser.
- D1 LocalDPO-style local corruption.
- D2 ProPainter loser.
- D3 self + ProPainter score mix.
- D4 defect-balanced medium-hard mix.

## Fallback Candidate: Stage-Aware Spatial/Temporal Preference Decomposition

Short name: `ST-Pref`

Core idea:

Use distinct preference signals for DiffuEraser/VideoPainter spatial and temporal stages. Stage1 focuses on hole and boundary realism. Stage2 focuses on temporal flicker, motion-boundary stability, and consistency.

Role:

Fallback if RC-FPO data/region results are positive but temporal consistency remains flat or degraded.

First experiments:

- Stage1 current LoVI vs RC-FPO data.
- Stage2 temporal-defect manifest vs reusing Stage1 data.

## Paused Candidate: Region-Conditioned SDPO

Use as an ablation:

- Exact SDPO.
- LoVI + global SDPO lambda.
- LoVI + region-conditioned heuristic lambda.

No primary claim unless new proof or strong unique evidence appears.

## Paused Candidate: Linear Utility / EMA

Use as an ablation:

- Linear-Frozen.
- Linear-EMA.

No primary claim because Linear-DPO already owns the objective idea.
