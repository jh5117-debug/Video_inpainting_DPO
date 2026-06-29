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

## 2026-06-29 Strict Visual Relabeling

Status: `MINIMAX_TARGETED_RELABEL_COMPLETED`.

Codex opened `47/47` selected review pages covering `369` auto success /
medium-hard rows, then relabeled all `452` candidates.

Final conservative label counts:

- `SUCCESS_CLEAN`: `33`
- `SUCCESS_USABLE`: `92`
- usable success including clean: `125`
- `FAILURE_MEDIUM_HARD`: `137`
- `FAILURE_BOUNDARY_BAD`: `21`
- `FAILURE_FOGGING`: `5`
- `FAILURE_OUTSIDE_BAD`: `50`
- `FAILURE_TOO_CLOSE`: `27`
- `BORDERLINE_REJECT`: `87`

Same-source precheck:

- groups with usable success and medium-hard failure: `10`
- one-to-one pair precheck: `18`
- capped same-source combination precheck: `40`

This unlocks Milestone D pair construction only. It does not unlock training,
bad-noise v4, Stage2 handoff, or a MiniMax quality-positive claim.
