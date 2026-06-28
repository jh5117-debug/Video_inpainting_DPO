# Exp31 VideoPainter 2000-Step Final Decision

Final status: `VIDEOPAINTER_2000_POSITIVE`

Step2000 has real long-run evidence on fixed search-dev and fixed shadow-dev.
Strict official-base identity passed, and the completed LPIPS / mask-region
Ewarp evaluation now satisfies the formal VideoPainter-only positive gate.

## Primary Findings

- Checkpoint audit passed for the explicit ladder through Step2000.
- Search-dev Step2000 vs Step0: full PSNR `+5.5701`, mask PSNR `+9.9747`, sampled boundary PSNR `+12.0920`, win rate `0.9688`.
- Search-dev Step2000 vs Step50: full PSNR `+6.1338`, mask PSNR `+1.8747`, sampled boundary PSNR `+3.7226`, sampled outside L1 `-10.0351`, win rate `1.0000`.
- Shadow-dev Step2000 vs Step0: full PSNR `+6.2632`, mask PSNR `+10.8860`, sampled boundary PSNR `+12.2343`, win rate `1.0000`.
- Shadow-dev Step2000 vs Step50: full PSNR `+6.4772`, mask PSNR `+2.0832`, sampled boundary PSNR `+3.9405`, sampled outside L1 `-10.5232`, win rate `1.0000`.
- Strict base identity audit: official base and Step0 weights match exactly;
  replay-vs-existing Step0/50/2000 raw/comp frames are exact on 2 search-dev +
  2 shadow-dev rows.
- LPIPS/Ewarp completion: `384/384` metric rows OK in
  `reports/exp31_vp_2000_lpips_ewarp_metrics.md`.
- Shadow-dev comp Step2000 vs Step0: full PSNR `+11.440561`, full LPIPS
  `-0.056840`, mask LPIPS `-0.213718`, boundary PSNR `+15.242894`, mask-region
  Ewarp `-11.171650`, probability improved `1.0000`.
- Shadow-dev comp Step2000 vs Step50: full PSNR `+2.305730`, full LPIPS
  `-0.008813`, mask LPIPS `-0.034082`, boundary PSNR `+3.637059`, mask-region
  Ewarp `-0.258536`, probability improved `>=0.9062`.

## Visual Review

- Opened search-dev and shadow-dev all-32 evidence/crop pages for Step0, Step50, and Step2000.
- Step0 is weak/noisy with poor local fill and boundary artifacts.
- Step50 improves the mask region but repeatedly introduces outside brightness/color pollution and purple/green local artifacts.
- Step2000 is visibly cleaner than Step50 and not visually collapsed; a minority of rows still show residual local texture or mild darkening.

## Decision Answers

1. VideoPainter now has 2000-step evidence: yes, formal VideoPainter-only
   long-run positive evidence on fixed search-dev and shadow-dev.
2. Step2000 improves over Step0: yes on available metrics and visual review.
3. Step2000 improves over Step50: yes on available metrics and visual review.
4. Step50 is not better than Step2000 on this evaluation.
5. The paper can report VideoPainter Step2000 as a 2000-step long-run positive
   under this fixed protocol, while preserving the earlier 50-step result as
   micro evidence.
6. New long-run artifacts are not systemic; finite residual texture/darkening remains.
7. This strengthens VideoPainter cross-backbone evidence, but not
   universal-adapter, final-SOTA, all-models, or top-conference novelty claims.

## Metric Caveats

- TC is recorded as `TC_BACKEND_NOT_LOCAL`; no automatic TC model download was
  triggered and no proxy is reported as real TC.
- Ewarp is mask-region Ewarp from the existing `inference/metrics.py` backend
  with OpenCV DIS fallback because RAFT weights were not local on PAI.
- Comp outside pixels are copied from the winner by protocol, so outside L1 is
  exactly `0.0` in the LPIPS/Ewarp completion and is not model-predicted outside
  preservation evidence.

## Formal Decision

The formal status is `VIDEOPAINTER_2000_POSITIVE` for VideoPainter only.
No universal adapter, final SOTA, all-models-supported, or top-conference
novelty claim is made.
