# Exp51 VOID Loser-Dominant Rescue

Created: 2026-07-01T10:33:01+08:00
Branch: `research/exp51-void-loser-dominant-rescue-20260701`
Base: Exp50 HEAD `3aa65c53b2bd53e69e7a2d9528d9127a21849d66`

## Objective

Diagnose Exp50 loser-dominant 10-step behavior and test winner-preserving / local-DPO / SDPO-safe rescue recipes without modifying VOID official source, shared trainer, or `inference/metrics.py`.

## Milestone A

Status: `VOID_LOSER_DOMINANT_CONFIRMED`

Exp50 10-step margin growth is dominated by loser degradation rather than winner improvement. VOID remains a baseline / loser generator / adapter engineering candidate, not third adapter evidence.

## Safety

No VOR-Eval, no hard comp, no long training, no universal adapter, and no final SOTA claims.
