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

## 2026-06-26 EffectErase Command Dry-Run

- Full inference: not run.
- Script import: passed in dedicated venv.
- Transformers: `4.51.3`.
- Diffusers: `0.31.0`.
- Decord: `0.6.0`.
- Asset SHA256 rechecked for LoRA, VAE, DiT, image encoder, and text encoder.
- Command readiness: `EFFECTERASE_COMMAND_READY`, but blocked from inference by
  `EFFECTERASE_SMOKE_INPUTS_BLOCKED`.

## 2026-06-26 MiniMax Expanded Source-Pool Plan

- Required sources: 96 or 128.
- Audit rows available: 64.
- Valid aligned rows: 63.
- Previous source32 rows excluded: 32.
- Remaining valid rows: 31.
- Remaining source type counts: REAL 23, BLENDER 8.
- Remaining mask bucket counts: small 5, medium 12, large 14.
- Generation status: blocked before inference because source-pool size is
  insufficient.

## 2026-06-26 Continuation V4 Readback

- No new quantitative metrics; no model inference ran.
- EffectErase old smoke manifest SHA256:
  `54fd62a97fa69f2f17590488136d426cee77de0ed02548c46a83d8818be2b137`.
- EffectErase old smoke inputs remain 5/6 ready, with
  `REAL_ENV249_00103_004_04` blocked by empty mask.
- MiniMax prior expanded source manifest SHA256:
  `bb31cfa5abd320dc88a5471036a3b2bb54b91257d3f65380dc43ecdf29c60929`.
- Right-side GPU work remains pending; no EffectErase inference or MiniMax
  generation was launched by readback.

## 2026-06-26 EffectErase Smoke V2 Pre-Registration

- Accepted rows: 6.
- Rejected old rows: 1.
- Replacement row: `REAL_ENV248_00118_005_03`.
- New manifest SHA256:
  `b16a0007a22f190bb7894a673092063efb5dd2eda26dbd53737cdc987d9d4f36`.
- REAL/BLENDER counts: 3/3.
- Mask bucket counts: small 2, medium 2, large 2.
- Preview-reviewed rows: 6/6.
- Inference metrics: not run yet.

## 2026-06-26 EffectErase Smoke V2 Input Materialization

- Materialized rows: 6.
- Ready rows: 6.
- Blocked rows: 0.
- Condition/winner/mask frame counts: 17/17/17 for all rows.
- Resolution: 832x480 for all rows.
- Mask non-empty frames: 17/17 for all rows.
- Inference metrics: not run yet.

## 2026-06-26 EffectErase Official Inference Smoke V2

- Attempted rows: 1.
- Successful rows: 0.
- Output videos: 0.
- First failure: missing `diffsynth` import; fixed by `PYTHONPATH`.
- Final failure: official pipeline defaulted to 81 frames internally, producing
  noise latent time dimension 21 while the locked 17-frame inputs produced time
  dimension 5.
- Metrics are unavailable because no output video was produced.

## 2026-06-26 MiniMax Full-VOR Source Audit

- Full metadata rows read: 57,751.
- Raw scene groups: 1,449.
- Valid candidate scene groups after excluding previous Exp29 MiniMax source32
  and EffectErase smoke rows: 1,417.
- Locked source candidates: 192.
- Selected source type counts: REAL 96, BLENDER 96.
- Manifest SHA256:
  `16e128282da110eeefd6cb56a517c8b6de82e42a5241c9b845e01315d9800f10`.
- Mask bucket/effect/motion quantitative labels are unavailable in the full
  metadata index and remain pending materialization.
- No model output, image/video metric, candidate classification, recipe, or
  training metric was produced by this audit.

## 2026-06-26 MiniMax Expanded Data-Yield Review V2

- Seed A candidates: 96.
- Conditional seed B near-miss candidates: 32.
- Classification counts across all attempts:
  - `MEDIUM_HARD_ELIGIBLE`: 24
  - `HARD_BUT_PLAUSIBLE`: 2
  - `TOO_CLOSE`: 14
  - `TRIVIAL_BAD`: 77
  - `TECHNICAL_INVALID`: 11
- Eligible unique scene groups after best-candidate merge: 26.
- Required for train16+heldout16: 32 scene-disjoint eligible groups.
- Train trace rows: 16.
- Heldout rows: 0.
- Result: `MINIMAX_EXPANDED_DATA_YIELD_INSUFFICIENT`.
- No optimizer recipe, 30-step micro, or training metrics were produced.

## 2026-06-27 Continuation V5 Readback

- No new inference, video metric, optimizer metric, or training metric was
  produced by this readback.
- PAI GPU checks: 2.
- GPUs with active compute process during checks: 0.
- Left CLI reserved GPUs from heartbeat files: GPU1-GPU4.
- EffectErase pending quantitative gate: official 81-frame smoke.
- MiniMax pending quantitative gate: top-up data-yield candidate review.

## 2026-06-27 EffectErase Official 81F Source Audit

- Candidate triplets audited: 24.
- Accepted by 81F/frame/mask rules: 24.
- Locked rows: 8.
- Rejected/reserve rows recorded: 16.
- Source type counts: REAL 5, BLENDER 3.
- Mask bucket counts: small 3, medium 3, large 2.
- Manifest SHA256:
  `706cb09286fd8528d7efbbb91eb89673a9ec7ce61b0047e6b3b2e8ea4c9b1fb3`.
- No EffectErase model output, image/video metric, adapter metric, or training
  metric was produced by this source audit.

## 2026-06-27 EffectErase Official 81F Input Materialization

- Rows: 8.
- Ready rows: 8.
- Blocked rows: 0.
- Condition/winner/mask frames: 81/81/81 for every locked row.
- Resolution: 832x480.
- VOR-Eval use: false.
- Training eligibility: false.
- No EffectErase model output, image/video metric, adapter metric, or training
  metric was produced by this materialization milestone.

## 2026-06-27 EffectErase Official 81F Command Validation

- Rows validated: 8.
- Assets ready: true.
- Inputs ready: true.
- Official help/import return code: 0.
- Command protocol: 81 frames, 832x480, seed 2025, CFG 1.0, 50 steps.
- No EffectErase model output, image/video metric, adapter metric, or training
  metric was produced by this command-validation milestone.
