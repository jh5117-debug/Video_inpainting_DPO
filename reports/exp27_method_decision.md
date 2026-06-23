# Exp27 Method Decision

## Primary Candidate

`RC-FPO`: Restoration-Critical Failure-Structured Preference Optimization.

The primary method is a data-plus-region framework:

1. construct task-native failure-structured preference pairs for BR and OR;
2. keep a faithful LocalDPO-style random/local corruption baseline;
3. optimize using restoration-critical region decomposition rather than a single generic local mask;
4. evaluate explicit defect vectors and anti-shortcut diagnostics.

## Fallback Candidate

`ST-Pref`: Stage-Aware Spatial/Temporal Preference Decomposition.

Use only if RC-FPO improves spatial metrics but fails temporal consistency or if Stage2 continues to show no TC/Ewarp gain.

## Baselines Required Before New Claims

- Current LoVI objective.
- Faithful LocalDPO-style inpainting/OR baseline.
- Exact Diffusion-SDPO.
- LoVI + SDPO heuristic, clearly labeled as heuristic.
- Linear-DPO Frozen.
- Linear-DPO EMA.

## Do Not Run Yet

No 500/1000/2000-step Exp27 training is allowed until:

- LocalDPO official mask issue is handled transparently;
- SDPO and Linear-DPO full-batch parity pass;
- Exp25 Gate128 OR loser generation is resolved;
- Exp26 Gate64 49F self-loser generation is complete;
- method-specific manifests are locked.

## Immediate Next Experiment

Use DiffuEraser BR Stage1 with equal source videos and equal budget to compare:

- D0 current task-native self loser;
- D1 LocalDPO-style local corruption loser;
- D2 ProPainter loser;
- D4 defect-balanced medium-hard mix.

Run only 1/10/50-step micro gates first.
