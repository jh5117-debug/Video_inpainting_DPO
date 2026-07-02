# Exp59 Final Decision

Status: `VOID_TARGET_HIT_WEAK_REGENERATE_DATA`

Exp59 completed the official VOID pass1 inference diagnostic on Exp58B Kubric-native Gate8. No training, preference forward, zero-gap, one-step, or 10-step was run.

## Required Answers

1. Is official VOID inference usable on Kubric Gate8?

Yes, technically. Official VOID pass1 ran on all 8 Kubric samples and produced 8/8 raw outputs, 8/8 tuple outputs, and 8/8 evidence packs.

2. Are Kubric outputs technically valid?

Yes. Metrics and visual review found 8/8 technically valid outputs. The controlled runtime fix was limited to a run-local static `imageio-ffmpeg` shim after system ffmpeg failed with missing `libblas.so.3`.

3. Does Kubric-native data behave better than VOR-derived data?

No clear evidence. Both are 8/8 technical valid and outside-safe. Kubric-native outputs have similar full PSNR and higher SSIM, but transition-region quality is weaker in this Gate8: overlap PSNR `16.673219`, affected PSNR `17.527094`, and boundary PSNR `22.267098`.

4. Is data mismatch confirmed, suspected, or inconclusive?

Data mismatch remains suspected but is not confirmed. Exp59 cannot isolate VOR-to-VOID data mismatch because all Kubric rows have `target_hit=false`.

5. Is `target_hit=false` a blocker for adapter data?

Yes. It makes this Gate8 usable for official inference diagnostics, but not for adapter training or one-step preference evidence.

6. Should we generate `target_hit=true` Gate8?

Yes. The next minimal experiment should regenerate Kubric Gate8 with target-hit-positive rows and preferably a larger or official-compatible resolution before any Kubric one-step.

7. Should we run Kubric one-step next?

No, not from this Gate8. Run Kubric one-step only after target-hit-positive native data passes input and official inference review.

8. Did VOID become third adapter evidence?

No. Exp59 contains no training or micro-gate pass.

9. Should VOID stay as baseline / loser generator / adapter candidate?

Yes. VOID remains a VOR-OR inference baseline, same-model loser generator candidate, and adapter-engineering candidate. It is not third-backbone evidence.

## Final Scientific Decision

`VOID_KUBRIC_INFERENCE_DIAGNOSTIC_DONE`

`VOID_TARGET_HIT_WEAK_REGENERATE_DATA`

`VOID_NATIVE_DATA_NOT_READY`

The diagnostic establishes that official VOID inference can run on native Kubric-formatted inputs, but the current native data is too weak to test adapter feasibility. The correct next step is data repair, not loss tuning or 10-step training.
