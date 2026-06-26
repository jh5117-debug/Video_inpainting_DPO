# Exp26 Post-Confirmation Sanity Audit

- status: `EXP26_POSTCONFIRMATION_SANITY_AUDIT_PASSED`
- created_utc: `2026-06-26T01:10:21Z`
- scope: read-only audit of the completed VideoPainter shadow-dev confirmation
- no retraining, no checkpoint reselection, no 100-step, no RC-FPO

## Identity Checks

| Item | Result |
| --- | --- |
| Step50 same trajectory as Step10/Step30 | PASS |
| Step0 is fixed official initialization | PASS |
| Step50 differs from Step0 | PASS |
| Step10/30/50 checkpoint trees differ from Step0 | PASS |
| fallback detected | NO |
| primary/search/shadow overlap | `0 / 0 / 0` |
| shadow-dev used for selection | NO |
| frame0-GT handled separately | PASS |
| primary metrics use frame1-48 | PASS |
| comp outside-mask semantics | PASS |
| unexpected winner/GT leakage | NO |
| raw outside preservation audited | PASS |
| 32/32 shadow videos reviewed | PASS |
| seed robustness | PASS, `3 / 3` seeds |

The checkpoint identity report records `PENDING_RUNTIME_PREFLIGHT` in the strict
load column because the audit file stores static tree identity. The same
Step0/10/30/50 checkpoints were subsequently loaded by the official
VideoPainter inference path to generate all shadow-dev outputs without fallback
or missing-output failure. This audit therefore treats the completed official
pipeline load and non-fallback output generation as the runtime identity check,
while leaving the static key-diff field unchanged.

## Shadow-Dev Primary Metrics

Primary comparison is fixed Step50 minus fixed Step0 on frame1-48 comp.

| Metric | Mean Delta | Win Rate | Bootstrap 95% CI | Probability Improved |
| --- | ---: | ---: | ---: | ---: |
| strict mask PSNR | `+5.186942` | `0.781250` | `[+2.781118, +7.818869]` | `1.000000` |
| boundary PSNR | `+12.175098` | `1.000000` | `[+10.184673, +14.212251]` | `1.000000` |
| LPIPS | `-0.040142` | `0.937500` | `[-0.052638, -0.028834]` | `1.000000` |
| Ewarp | `-8.378847` | `0.968750` | `[-13.581173, -4.369433]` | `1.000000` |

Whole comp PSNR improved by `+5.160739` dB on frame1-48.

## TC / VFID

No-first-frame comp:

- Step0 TC: `0.986760`
- Step50 TC: `0.991139`
- Step0 VFID/FVD-style: `0.531078`
- Step50 VFID/FVD-style: `0.499650`

## Leakage And Video Review

- leakage rows audited: `128`
- non-expected GT leakage flags: `0`
- reviewer-pass rows: `32 / 32`
- Step50 clearly better: `12`
- Step50 slightly better: `13`
- tie: `3`
- Step0 slightly better: `1`
- Step50 new artifact: `3`

Review conclusion remains unchanged: Step50 often removes translucent oval/ring
residuals from Step0, with remaining local failures in water, grass, foliage,
and color-smear cases. No systematic frame-order bug, first-frame failure,
mask overwrite, or global collapse was found.

## Dynamics

- max grad norm: `471.683581`
- p95 grad norm: `136.259904`
- grad > 10 / 50 / 100: `41 / 7 / 4`
- implicit accuracy mean / last: `0.28 / 0.0`
- loser-dominant mean / last: `0.12 / 0.0`
- recommendation: `NO_100STEP_BY_PROTOCOL`

Large gradients did not correspond to Step50 quality collapse on shadow-dev.
They also do not authorize continuing to 100-step in this round.

## Decision

`EXP26_POSTCONFIRMATION_SANITY_AUDIT_PASSED`.

The existing shadow-dev confirmation is internally consistent and can be used as
post-confirmation evidence for VideoPainter on the locked VOR-BG distribution.
It still does not justify universal-adapter, final-SOTA, or top-conference
novelty claims; external clean-source validation remains the next scientific
check.

