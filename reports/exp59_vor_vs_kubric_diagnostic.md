# Exp59 VOR-vs-Kubric Diagnostic

Status: `VOID_TARGET_HIT_WEAK_NATIVE_DATA_INCONCLUSIVE`

Exp59 compared prior VOR-derived VOID diagnostics with the new Exp58B Kubric-native Gate8 official inference outputs. No training, one-step, or 10-step was run.

## Summary Table

| Field | VOR-derived Gate8 | Kubric-native Gate8 | Interpretation |
| --- | --- | --- | --- |
| technical valid | 8/8 | 8/8 | Official VOID inference works on both input families. |
| full PSNR | 30.174910 | 30.152555 | Similar aggregate full-frame score, but not sufficient for adapter readiness. |
| SSIM | 0.824404 | 0.919492 | Kubric synthetic low-res outputs have high structural similarity. |
| object/mask PSNR | 25.538001 | 28.337691 | Kubric object region is not worse by aggregate PSNR, but visual target residual remains. |
| overlap PSNR | NA | 16.673219 | Kubric exposes weak overlap performance directly. |
| affected PSNR | 25.764115 | 17.527094 | Kubric affected quality is weaker in this Gate8. |
| boundary PSNR | 25.843463 | 22.267098 | Kubric boundary/transition quality does not improve over VOR. |
| outside PSNR | 33.109107 | 34.210532 | Outside preservation is safe in both. |
| usable / bounded loser | 6/8 | 2/8 medium-hard, 2/8 too-close | Kubric Gate8 gives fewer useful loser candidates. |
| target_hit=false | NA | 8/8 | This weakens any native-data adapter conclusion. |

## Required Answers

1. **Does Kubric-native input reduce transition-region damage?** No clear evidence. Kubric overlap PSNR is `16.673219`, affected PSNR is `17.527094`, and visual review marks transition residual/damage in 6/8 rows.
2. **Does official VOID behave more naturally on Kubric-native than on VOR-derived data?** It runs cleanly and preserves outside/background, but it does not produce a stronger usable-loser pool than VOR Gate8 under this target-hit-false Gate8.
3. **Does target_hit=false make Gate8 insufficient?** Yes. All 8 rows carry `target_hit=false`, so the generated counterfactual target is weak for adapter data even though it is valid for inference diagnostics.
4. **Is VOR-to-VOID quadmask likely the blocker?** Still suspected from Exp50-Exp57, but not confirmed here. This Kubric Gate8 is too weak to isolate data mismatch from target-generation weakness.
5. **Or is wrapper/objective still likely the blocker?** Previous VOR one-step/rescue evidence still points to objective/transition-region blockers; Exp59 does not clear that blocker.
6. **Next step.** Regenerate Kubric Gate8 with `target_hit=true` and preferably at a larger/official-compatible resolution before any Kubric one-step. Do not run 10-step.

## Decision

Kubric-native official inference is technically usable, but this Gate8 does not confirm data mismatch and is not ready for adapter training. The strongest decision is `VOID_TARGET_HIT_WEAK_NATIVE_DATA_INCONCLUSIVE`: data mismatch remains suspected, target-hit quality must be repaired, and VOID remains a baseline / same-model loser generator / adapter-engineering candidate only.
