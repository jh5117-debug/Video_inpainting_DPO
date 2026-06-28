# Exp40 Status

Current status: `MINIMAX_SFT_PSNRSAFE_NEGATIVE`

## 2026-06-28 Readback

- Branch: `research/exp40-minimax-psnr-safe-rescue-20260628`.
- Start HEAD: `06b17c0a4be2cb82d1ffbdf7b6c93406f37a3ff8`.
- Base: `origin/research/exp38-minimax-full-adapter-breakthrough-20260628`.
- Exp38 PRD, registry, reports, metrics, and visual review were read.
- MiniMax remains plumbing-positive but not quality-positive.
- R1 is the only useful signal, but it is blocked by boundary/outside cost and
  visual tradeoff.
- GPU0/GPU1 were audited and a stale Exp30 GPU0 heartbeat process group was
  terminated after recording PID/PGID/cwd/cmdline. GPU2-GPU7 were untouched.
- No GPU training, inference, or new model output was launched in this
  milestone.

Next status target: `MINIMAX_R1_SIGNAL_AUDITED`.

## 2026-06-28 R1 Sample-Level Diagnosis

Current status: `MINIMAX_R1_SIGNAL_AUDITED`

- No new training, inference, or GPU task was launched.
- Exp38 SFT/DPO R1 only has heldout13 outputs.
- Existing Exp38 train-overfit Exp37 R1 outputs provide train32/heldout16
  context.
- Main blocker: boundary/outside cost plus fogging/over-erasure risk.
- DPO should not run before PSNR-safe SFT establishes safe search/shadow
  improvement.

Reports:

- `reports/exp40_r1_sample_level_diagnosis.md`
- `reports/exp40_r1_sample_level_diagnosis.csv`
- `reports/exp40_r1_visual_review.csv`
- `reports/exp40_r1_diagnosis_summary.json`

## 2026-06-29 LocalDPO v3 Pool

Current status: `MINIMAX_LOCALDPO_V3_POOL_READY_MINIMUM`

- No MiniMax training, inference, or adapter evaluation was launched.
- VOR-Train only; VOR-Eval was not used.
- Hard comp was not used.
- Selected pool reached the minimum, not the full target:
  `train=64`, `search=24`, `shadow=24`.
- Scene overlap is zero for train/search, train/shadow, and search/shadow.
- Selected rows are all `MEDIUM_HARD_ELIGIBLE`.
- Codex opened all 19 temporal-strip review pages covering the selected 112
  rows. This is a data-pool review, not a model-quality pass.
- PAI `hj` could not write to the requested experiments output root, so outputs
  are under the Exp40 log root.

Next status target: `MINIMAX_STEP0_BASELINE_READY`.

## 2026-06-29 Step0 Baseline

Current status: `MINIMAX_STEP0_BASELINE_ESTABLISHED`

- Ran Step0 MiniMax official baseline on the locked LocalDPO v3 minimum pool:
  `train=64`, `search=24`, `shadow=24`.
- Inference only: no training, DPO, 30-step, 100-step, hard comp, or VOR-Eval.
- Raw output is primary; diagnostic comp was not used.
- GPU0 ran train64; GPU1 ran search24+shadow24. GPU2-GPU7 were untouched.
- Aggregate raw metrics:
  - train full/mask/boundary/outside PSNR:
    `23.965598` / `18.485359` / `19.395954` / `26.458319`
  - search full/mask/boundary/outside PSNR:
    `25.043807` / `20.493872` / `21.409812` / `27.765446`
  - shadow full/mask/boundary/outside PSNR:
    `26.209732` / `21.645338` / `24.277694` / `29.577002`
- Codex opened 42 review pages: 14 midframe pages and 28 temporal-strip pages
  covering all 112 rows.
- This milestone establishes the baseline only; it is not a positive model
  result and does not unlock DPO.

Next status target: `MINIMAX_SFT_PSNRSAFE_30STEP_GATE`.

## 2026-06-29 PSNR-Safe SFT 30-Step Grid

Current status: `MINIMAX_SFT_PSNRSAFE_NEGATIVE`

- Ran the winner-SFT-only 30-step grid on GPU0/GPU1:
  - GPU0: `SFT-A`, `SFT-B`
  - GPU1: `SFT-C`, `SFT-D`
- Scope: `S0`.
- LR grid: `3e-5`, `1e-4`, `3e-4`.
- Search rows evaluated: `288` recipe-row outputs.
- Raw output was primary; hard comp and VOR-Eval were not used.
- GPU2-GPU7 were untouched; no PID/PGID needed to be killed for this milestone.
- No OOM/CUDA/Xid was observed.
- All 12 recipe aggregates failed the search numeric gate.
- Best aggregate recipe: `SFTmC_S0_lr3em05`, still negative:
  full/mask/boundary/outside `-1.816781` / `-1.634597` /
  `-1.899575` / `-2.624405`.
- Codex opened representative best/worst strips. High-LR recipes show obvious
  noisy/color collapse; low-LR recipes still have negative aggregate PSNR and
  unsafe boundary/outside tradeoff.

Decision:

- `MINIMAX_SFT_PSNRSAFE_NEGATIVE`
- no 100-step
- no DPO-after-SFT
- no 300/500-step confirmation
- MiniMax remains `MINIMAX_PLUMBING_POSITIVE_RECIPE_NOT_READY`

## 2026-06-29 Paper Positioning

Current paper status: `TWO_BACKBONE_PLUS_MINIMAX_PLUMBING_ONLY`

- DiffuEraser + VideoPainter remain the confirmed positive adapter evidence.
- MiniMax remains plumbing-positive and trainable, but Exp40 does not provide
  third-backbone adapter-positive evidence.
- Universal-adapter / all-models-supported / final-SOTA language remains
  forbidden.
