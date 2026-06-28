# Exp40 Status

Current status: `EXP40_READBACK_COMPLETED`

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
