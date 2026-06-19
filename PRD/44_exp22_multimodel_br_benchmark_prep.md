# PRD 44: Exp22 Multimodel BR Benchmark Prep

Date: 2026-06-19

## Scope

Exp22 prepares public models for real-weight BR inference smoke and unified BR
baseline outputs. It does not run OR training, OR sweep, or formal cross-model
DPO quality experiments.

## Status

```text
ASSET_SCAN_SCAFFOLD_READY
```

The asset matrix scanner is implemented. Missing gated data/weights are reported
as blockers and are not bypassed.

## Models

- DiffuEraser
- FloED
- CoCoCo
- VideoComposer / VideoComp
- VACE
- MiniMax-Remover
- EffectErase
- ProPainter
- VideoPainter

## EffectErase

VOR / VOR-Eval / VOR-Wild data remains:

```text
WAITING_AUTH
```
