# Exp59 VOID Kubric Gate8 Official Inference Diagnostic

Date: 2026-07-02

Branch: `research/exp59-void-kubric-gate8-inference-20260702`

Base: `origin/research/exp58-void-native-data-diagnostic-20260702`

## Goal

Exp59 tests the newly generated Exp58B VOID-native Kubric Gate8 data with official VOID pass1 inference. It is a diagnostic of native Kubric Step0/baseline behavior and a VOR-derived versus Kubric-native failure-pattern comparison.

## Current Gate

Current status: `VOID_TARGET_HIT_WEAK_REGENERATE_DATA`

The Exp58B Gate8 manifest exists and contains 8 valid native Kubric rows. PAI decoded all 8 `rgb_full`, `rgb_removed`, and quadmask videos as 24-frame, 128x128, 8 fps clips with quadmask values `0|63|127|255`. The data is weak for adapter training because all metadata rows report `target_hit=false`, but it is sufficient for official inference diagnostics.

Milestone B confirmed that official pass1 inference can proceed through the normal `predict_v2v.py` preprocessing path. The 128x128 inputs will be resized to the official `384x672` sample size and padded to the 85-frame temporal window. This is acceptable for diagnostic inference but must be recorded when computing metrics against the native 128x128 `rgb_removed` ground truth.

Milestone C materialized all 8 official input folders. The configured experiment output root under `/experiments/dpo` is not writable on PAI, so Exp59 will use the writable `/logs/autoresearch` and `/runtime` roots for official outputs while recording the fallback.

Milestone D completed official VOID pass1 inference on 8/8 Kubric samples. The controlled runtime fix used a run-local `imageio-ffmpeg` binary after the first attempt failed on system `/usr/bin/ffmpeg` missing `libblas.so.3`. The final run produced 8 raw outputs, 8 tuple outputs, and 8 evidence packs.

Milestone E completed native-space metrics and visual review. Metrics compare the first 24 output frames after downscaling the official `384x672` output to native `128x128` against `rgb_removed`. Mean metrics: full PSNR `30.152555`, SSIM `0.919492`, object PSNR `28.337691`, overlap PSNR `16.673219`, affected PSNR `17.527094`, boundary PSNR `22.267098`, and outside PSNR `34.210532`. Codex opened all 8 review contact sheets. All outputs are technically valid and outside/background is stable, but all rows remain `target_hit=false`; only 2/8 are medium-hard loser diagnostics, 2/8 are too-close/weak, and 6/8 show transition residual/damage. This is a useful official-inference diagnostic, not adapter evidence.

Milestone F compared Exp59 Kubric-native inference with Exp50 VOR-derived Gate8. Both are 8/8 technical valid and outside-safe. Kubric does not show reduced transition-region damage in this Gate8: overlap PSNR `16.673219`, affected PSNR `17.527094`, and boundary PSNR `22.267098`, versus VOR affected/boundary means near `25.8`. The result is inconclusive for data-mismatch confirmation because all Kubric rows have `target_hit=false`. Next minimal experiment is to regenerate target-hit-positive, preferably higher-resolution Kubric Gate8 before any Kubric one-step.

Final decision: official VOID inference is usable on Kubric Gate8, but this target-hit-false native data is not ready for adapter training. One-step and 10-step remain locked. Next minimal experiment is target-hit-positive Kubric Gate8 regeneration followed by the same official inference review.

## Scope

Allowed:

- materialize official VOID inference inputs from Kubric Gate8
- run official VOID pass1 inference only
- compute full and quadmask-aware metrics against `rgb_removed`
- generate and inspect visual evidence
- compare Kubric-native output patterns against prior VOR-derived VOID diagnostics

Forbidden:

- training
- preference forward
- zero-gap
- one-step
- 10-step
- long training
- VOR-Eval tuning
- hard comp
- modifying VOID official source
- modifying `inference/metrics.py`
- modifying shared trainer
- universal-adapter or final-SOTA claims
- third-backbone evidence claims

## Storage

Large inputs and outputs stay on PAI/NAS:

- input root: `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/void/kubric_exp58b/gate8/gate8`
- output root: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp59_void_kubric_gate8_inference`
- log root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp59_void_kubric_gate8_inference`
- runtime root: `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp59_void_kubric_gate8_inference`

HAL is used only for git/report control unless PAI is unavailable.

## Milestones

1. Read back Gate8 and audit inputs.
2. Audit official VOID inference protocol.
3. Materialize official input folders.
4. Run official VOID pass1 inference on exactly 8 Kubric samples.
5. Compute metrics and perform visual review.
6. Compare VOR-derived and Kubric-native failure patterns.
7. Write final diagnostic decision.

## Claim Boundary

VOID remains an inference baseline, same-model loser generator candidate, and adapter-engineering candidate. Exp59 cannot make VOID a third adapter/backbone evidence because no training or micro-gate is run.
