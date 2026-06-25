# Exp27 Metric Summary

## True SDPO Gate

Status: `TRUE_MODEL_PARITY`

- S1 records: `128`
- S1 `lambda_safe < 1`: `32 / 128 = 0.25`
- SDPO lambda max abs diff: `0.0`
- SDPO loss max abs diff: `0.0`
- output-gradient cosine min: `0.9999998807907104`
- tiny-step reference grad norm max: `0.0`

## True Linear-DPO 1/10-Step Gate

Status: `LINEAR_TRUE_MODEL_1_10_STEP_PASSED`

| Variant | Records | Max Grad Norm | Step10 Loss | Policy Delta | Reference Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear-Frozen | 10 | 0.48048678696353975 | -0.1105356365442276 | 0.0012969709135074255 | 0.0 |
| Linear-EMA | 10 | 0.49775458360972635 | -0.11877599358558655 | 0.0013002275364513564 | 1.819002953296671e-08 |

No RC-FPO, 50-step objective study, or video-quality-positive claim has been
started from these technical gates.
