# Exp31 VideoPainter 2000-Step Final Decision

Final status: `VIDEOPAINTER_2000_PARETO_MIXED`

Step2000 has real long-run evidence on fixed search-dev and fixed shadow-dev. The available quantitative metrics and reviewed videos both favor Step2000 over Step0 and Step50, but the formal `VIDEOPAINTER_2000_POSITIVE` gate is not fully satisfied because LPIPS and Ewarp were not computed in this fast summary.

## Primary Findings

- Checkpoint audit passed for the explicit ladder through Step2000.
- Search-dev Step2000 vs Step0: full PSNR `+5.5701`, mask PSNR `+9.9747`, sampled boundary PSNR `+12.0920`, win rate `0.9688`.
- Search-dev Step2000 vs Step50: full PSNR `+6.1338`, mask PSNR `+1.8747`, sampled boundary PSNR `+3.7226`, sampled outside L1 `-10.0351`, win rate `1.0000`.
- Shadow-dev Step2000 vs Step0: full PSNR `+6.2632`, mask PSNR `+10.8860`, sampled boundary PSNR `+12.2343`, win rate `1.0000`.
- Shadow-dev Step2000 vs Step50: full PSNR `+6.4772`, mask PSNR `+2.0832`, sampled boundary PSNR `+3.9405`, sampled outside L1 `-10.5232`, win rate `1.0000`.

## Visual Review

- Opened search-dev and shadow-dev all-32 evidence/crop pages for Step0, Step50, and Step2000.
- Step0 is weak/noisy with poor local fill and boundary artifacts.
- Step50 improves the mask region but repeatedly introduces outside brightness/color pollution and purple/green local artifacts.
- Step2000 is visibly cleaner than Step50 and not visually collapsed; a minority of rows still show residual local texture or mild darkening.

## Decision Answers

1. VideoPainter now has 2000-step evidence: yes, but qualified.
2. Step2000 improves over Step0: yes on available metrics and visual review.
3. Step2000 improves over Step50: yes on available metrics and visual review.
4. Step50 is not better than Step2000 on this evaluation.
5. The paper should report VideoPainter as qualified 2000-step long-run evidence, while preserving the earlier 50-step micro evidence.
6. New long-run artifacts are not systemic; finite residual texture/darkening remains.
7. This strengthens cross-backbone adapter evidence, but not universal-adapter, final-SOTA, all-models, or top-conference novelty claims.

## Formal Blocker

LPIPS and Ewarp were not computed in this fast summary. Because the prompt requires those metrics for `VIDEOPAINTER_2000_POSITIVE`, the formal status is conservatively `VIDEOPAINTER_2000_PARETO_MIXED`.
