# Exp44 Metric Summary

Readback imports Exp42 counts:

- success rows: `52`
- failure rows: `80`
- success scene groups: `18`
- failure scene groups: `29`
- same-source overlap: `7`

Exp44 target: raise same-source usable pairs to at least `24`, ideally `48`.

## 2026-06-29 Targeted Source Manifest

- Source rows prepared for mining: `40`
- Candidate seed budget: `452`
- Missing source rows: `0`
- Fallback groups included: `false`

## 2026-06-29 Targeted Second-Pass Mining

Status: `MINIMAX_TARGETED_MINING_COMPLETED`.

Automatic label counts from official MiniMax raw inference:

- candidates: `452`
- technical failures: `0`
- successful-removal candidates: `138`
- medium-hard failure candidates: `231`
- boundary-bad candidates: `31`
- fogging/over-erasure candidates: `25`
- too-close candidates: `27`
- same-source pair capacity before visual relabel: `26`
- overlap groups before visual relabel: `13`

Aggregate metric means:

- full PSNR: `29.195214`
- mask PSNR: `23.476755`
- boundary PSNR: `24.611562`
- outside PSNR: `31.680892`
- outside MAE: `4.717091`
- temporal diff MAE: `1.533492`

These labels are provisional. Formal pair construction still requires strict
visual relabeling.
