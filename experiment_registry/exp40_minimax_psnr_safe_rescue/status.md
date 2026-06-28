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
