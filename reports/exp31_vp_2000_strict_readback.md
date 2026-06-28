# Exp31 VideoPainter 2000 Strict Validation Readback

Date: 2026-06-28

Status: `VIDEOPAINTER_2000_STRICT_READBACK_COMPLETE_BASE_AUDIT_PENDING`

This is a readback-only milestone. It does not start training, does not launch
MiniMax, does not modify `inference/metrics.py`, and does not modify shared
trainer code.

## Git And Protection

- branch: `research/exp31-videopainter-2000step-longrun-20260627`
- readback HEAD: `e8c068db5f3ec7b807af86e0dfab2a64431f8f37`
- current committed decision before this readback: `VIDEOPAINTER_2000_PARETO_MIXED`
- right-side MiniMax / Exp36: read-only protection remains active; GPU0 is not
  used by this lane.

## Source Identity

Exp31 is based on `origin/research/exp26-videopainter-dpo-v2` at base HEAD
`568a7dfb48bcdfce893176a1dd48c653414a13a8`.

Exp26 fixed 50-step identity:

- run: `vp_primary32_50step_20260625_171032`
- primary32 SHA256:
  `82f661f2f30a581a213972533817624217eabb97eba7aaeedc00ee2109e4e716`
- search-dev SHA256:
  `41c6571d26e4a5130818dd50fbbe1314c9d953284561a3cd20f630572f7c2a71`
- shadow-dev SHA256:
  `0338dba1513cfe0e5dd85cbf793b3782902b981ac9610b0e472c6a048f738c02`

Exp31 training and evaluation readback:

- training run: `exp31_vp2000_fresh_step0_20260627_133831`
- training policy: fresh from Step0, not a continuation of Exp26 Step50.
- explicit checkpoints: `0,1,10,50,100,200,500,1000,1500,2000`
- evaluation run: `exp31_vp2000_eval_step0_50_2000_20260628_032700`
- evaluated checkpoints: `step0`, `step50`, `step2000`
- evaluated splits: fixed `search-dev` and fixed `shadow-dev`
- all 6 generation/review groups completed `32/32` rows.

## Protocol Readback

The Exp31 evaluation controller used:

- VideoPainter root:
  `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter`
- official base:
  `ckpt/CogVideoX-5b-I2V`
- generation script:
  `exp26_videopainter_dpo_v2/code/run_vp2_gate64_official_generation.py`
- review script:
  `exp26_videopainter_dpo_v2/code/review_gate64_official_outputs.py`
- resolution: `720x480`
- frames: formal VideoPainter `49F`
- inference steps: `20`
- guidance scale: `6.0`
- seed: `20260627`
- dtype: `bf16`
- first-frame-GT handling: enabled through mask-zeroing on frame 0 when the
  row has `first_frame_gt=true`.

The generation script reads masks with polarity `mask > 127` means inpaint /
hole. Conditioning is built as the winner frame with the mask region zeroed.
The diagnostic comp formula in code is:

`comp = raw inside mask + winner outside mask`

where the mask is resized by nearest-neighbor if needed. This matches the
expected diagnostic comp policy; it is not hard-comp training input.

## Required Answers

1. Was Step2000 trained from the same base as Step50?

Yes at the source/config level. Exp31 was created from the Exp26 VideoPainter
branch and uses the same official VideoPainter base path and trainer family.
However, Exp31 Step2000 is a fresh Step0-to-2000 run, not a continuation of
Exp26 Step50. Milestone B must still verify checkpoint/base identity at the
file and replay-output level.

2. Is Step0 official VideoPainter base?

Expected yes: checkpoint-0 is saved before optimizer updates from the same
official base/branch loader. Formal promotion still requires Milestone B
replay: official base replay and Step0 replay must match the existing Step0
outputs or the result becomes `VIDEOPAINTER_BASELINE_IDENTITY_BLOCKED`.

3. Are Step50 and Step2000 evaluated on the same search/shadow rows?

Yes. Exp31 Step0/50/2000 all use the same symlinked mask-ready manifests:

- search:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/searchdev_step0_official_20260625_131957/gate64_mask_ready.jsonl`
- shadow:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/shadowdev_confirmatory_20260625/gate64_mask_ready.jsonl`

Both manifests have 32 rows and match the locked Exp26 search/shadow split
identity.

4. Are mask polarity, comp formula, frame count, resolution, seed, and
first-frame-GT identical?

Readback says yes for the Exp31 Step0/50/2000 evaluation path:

- mask polarity: `255 = inpaint / hole`, `0 = keep`
- comp formula: `raw inside mask + winner outside mask`
- frame count: `49`
- resolution: `720x480`
- seed: `20260627`
- first-frame-GT: frame-0 mask is forced to zero when enabled

Milestone B must still validate by replay and direct output comparison.

5. Why are gains so large?

The current evidence suggests Step50 learned the masked region but introduced
substantial outside/color pollution on this long-run split, while Step2000
substantially reduced that pollution and improved full-frame, mask, and sampled
boundary PSNR. The gains are also large because Step0 is the untrained official
base under this VOR-BG object-removal protocol and is visibly weak/noisy in the
masked region. This is plausible, but the size of the gains requires base
identity and replay audit before formal promotion.

6. What metrics are missing?

The committed fast summary is missing reliable:

- LPIPS
- Ewarp
- mask LPIPS
- boundary LPIPS
- raw-vs-comp metric separation under the Exp26 official metric backend
- bootstrap CI and leave-one-out sensitivity for LPIPS/Ewarp

7. What exact checks are required before formal positive?

Before `VIDEOPAINTER_2000_POSITIVE` is allowed:

- official base / Step0 / Step50 / Step2000 checkpoint identity audit passes;
- deterministic replay for 2 search and 2 shadow samples passes or is explained
  within deterministic tolerance;
- comp formula and mask polarity audit passes;
- no GT leakage or first-frame-GT protocol mismatch is found;
- LPIPS and Ewarp are computed with the same implementation/preprocessing used
  for Exp26 where possible;
- Step2000 vs Step0 shadow-dev improves PSNR or local metrics clearly;
- Step2000 vs Step50 shadow-dev is not worse overall;
- LPIPS is not worse by more than `0.001`;
- Ewarp is not worse by more than `0.05`;
- outside preservation is not systematically worse;
- visual better rate is at least `50%`;
- new artifact rate is at most `25%`.

Until those checks pass, the correct status remains
`VIDEOPAINTER_2000_PARETO_MIXED`.
