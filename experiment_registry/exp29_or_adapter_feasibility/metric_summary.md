# Exp29 Metric Summary

No metric-producing smoke or adapter gate has run yet.

MiniMax has verified PAI/NAS weights and is eligible for isolated inference
smoke. EffectErase is blocked before metrics because official weights were not
found.

## 2026-06-26 MiniMax Trainable Forward

- no-grad loss: 0.0171425510
- grad loss: 0.0171425510
- grad norm: 0.7473063172
- gradient tensors: 461
- peak VRAM: 12561.50 MiB
- missing keys: 0
- unexpected keys: 0

## 2026-06-26 MiniMax Adapter Gates

- zero-gap DPO loss: 0.6931471825
- one-step grad norm preclip: 0.8897291490
- one-step parameter delta probe: 2.0694979922992497e-11
- step10 parameter delta probe: 1.1061271569642785e-10
- step10 peak VRAM: 44614.22 MiB
- heldout `davis_hockey` PSNR delta: -0.0008006331
- heldout `davis_koala` PSNR delta: +0.0024137239

## 2026-06-26 MiniMax 10-Step Failure Analysis

- successful recovery optimizer: `SGD(lr=1e-7)`
- one-step parameter delta probe: `2.0694979922992497e-11`
- step10 parameter delta probe: `1.1061271569642785e-10`
- reference delta probe: `0.0`
- DPO loss range over 10 steps: `0.6931452155` to `0.6931496859`
- preference margin range: `-5.0142407417e-06` to `3.9562582970e-06`
- mean preclip grad norm: `0.7237282794`
- max preclip grad norm: `1.2341757971`

Metric conclusion: the 10-step run had finite gradients and strict reloads, but
the stable update was too small to produce a meaningful heldout output change.

## 2026-06-26 MiniMax Preference Data Quality Gate

- Candidates: `96/96`.
- Classification counts: 23 `MEDIUM_HARD_ELIGIBLE`, 4
  `HARD_BUT_PLAUSIBLE`, 3 `TOO_CLOSE`, 60 `TRIVIAL_BAD`, 6
  `TECHNICAL_INVALID`.
- Eligible candidates: 27.
- Eligible unique source groups: 9.
- Split result: train rows 9, heldout rows 0; no scene-disjoint train16 and
  heldout16 split can be formed.

## 2026-06-26 EffectErase Weight Recovery

- SHA256 manifest entries checked: 19.
- SHA256 result: all `OK`.
- Cache size: 20G.
- File count: 53.
- Key assets recovered: `EffectErase.ckpt`, `Wan2.1_VAE.pth`,
  `diffusion_pytorch_model.safetensors`,
  `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`,
  `models_t5_umt5-xxl-enc-bf16.pth`.

## 2026-06-26 Architecture Family Audit

No metric-producing task ran. The audit confirms MiniMax gates use a native
flow velocity target (`epsilon - z0`) rather than DiffuEraser epsilon DPO, so
`MINIMAX_GATE_INVALID_TARGET_MISMATCH` is not triggered.

## 2026-06-26 MiniMax Preference Data Quality Gate

- source rows: 32
- seeds per source: 3
- generated candidates: 96
- `MEDIUM_HARD_ELIGIBLE`: 23
- `HARD_BUT_PLAUSIBLE`: 4
- `TOO_CLOSE`: 3
- `TRIVIAL_BAD`: 60
- `TECHNICAL_INVALID`: 6
- eligible candidates: 27
- eligible unique scene groups: 9
- train manifest rows: 9
- heldout manifest rows: 0

Metric conclusion: the candidate yield is not sufficient for a fair
scene-disjoint train16/heldout16 MiniMax micro gate.

## 2026-06-26 EffectErase Smoke Pre-Registration

- No new quantitative metrics; inference has not run.
- Locked diagnostic smoke rows: 6.
- Manifest SHA256:
  `54fd62a97fa69f2f17590488136d426cee77de0ed02548c46a83d8818be2b137`.
- Balance: REAL/BLENDER = 3/3; small/medium/large masks = 2/2/2.
- Fixed protocol: 17 frames, 832x480, seed 2025, CFG 1.0, 50 steps.

## 2026-06-26 Continuation V3 Readback

- No new quantitative metrics; no model inference ran.
- EffectErase smoke manifest remains locked at SHA256
  `54fd62a97fa69f2f17590488136d426cee77de0ed02548c46a83d8818be2b137`.
- GPU readback found no active compute processes, but GPU1-GPU4 remain
  reserved by left CLI runtime locks.

## 2026-06-26 EffectErase Smoke Input Materialization

- Rows checked: 6.
- Ready input rows: 5.
- Blocked input rows: 1.
- All decoded ready videos have 17 frames at 832x480.
- Blocked row: `REAL_ENV249_00103_004_04`.
- Blocker metric: mask area min/mean/max all `0.0`.
