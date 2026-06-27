# Exp30 Three-Backbone Paper Positioning

Date: 2026-06-27

Status: `EXP30_THREE_BACKBONE_POSITIONING_LOCKED`

## Purpose

This report locks paper language before Exp30 starts multi-model VOR-OR pool
generation. It prevents EffectErase baseline evidence, MiniMax plumbing
evidence, and VideoPainter VOR-BG evidence from being over-claimed.

## DiffuEraser

- Backend family: SD / UNet-style latent diffusion with DiffuEraser-specific
  video inpainting plumbing.
- Current evidence: the project has DiffuEraser BR LoVI-DPO success in the
  main lineage.
- Exp30 gap: DiffuEraser still needs VOR-OR Stage1/Stage2 micro evidence before
  it can be used as VOR-OR paper evidence.
- Paper role: primary original backbone and the VOR-OR adapter baseline to be
  validated in Exp30 micro experiments.

## VideoPainter

- Backend family: VideoPainter / CogVideoX-style 49-frame video model.
- Current evidence: search-dev positive and independent shadow-dev confirmed.
- External evidence: external DAVIS-derived validation was not confirmed.
- Paper role: second backbone showing cross-backbone BR/inpainting adapter
  evidence.
- Important boundary: VideoPainter uses VOR-BG clean videos for BR/self-loser
  inpainting. It is not a standard VOR-OR adapter result.

## MiniMax-Remover

- Backend family: Wan2.1 / DiT / flow-matching OR model.
- Target: velocity `v = epsilon - z0`.
- Current evidence: repo/weights ready, inference smoke, trainable forward,
  zero-gap, and one-step strict reload passed.
- Current blocker: MiniMax-only loser mining is data-yield-limited. Exp29 found
  only 26 eligible unique scene groups, below the 32 needed for a fair
  train16+heldout16 split.
- Paper role: flow-style third-backbone adapter candidate and the main adapter
  target of Exp30.

## EffectErase

- Backend family: Wan / DiT-style official removal pipeline.
- Current evidence: official 81-frame OR baseline diagnostic passed 8/8 rows.
- Adapter audit: official removal pipeline has removal-specific adapters/task
  tokens, but no removal-specific training loss. Generic Wan training is not an
  acceptable replacement.
- Paper role: OR strong baseline / diagnostic / upper reference.
- Boundary: EffectErase is not adapter evidence in Exp30.

## Proposed Paper Language

Allowed:

- Cross-backbone adapter evidence on DiffuEraser plus VideoPainter.
- MiniMax as a flow-style third-backbone adapter candidate.
- EffectErase as a strong OR baseline and diagnostic reference.
- Model-specific backend adapters rather than a universal adapter.

Conditionally allowed if Exp30 MiniMax passes:

- DiffuEraser, VideoPainter, and MiniMax demonstrate multi-backbone adapter
  feasibility across SD/UNet, CogVideoX-style, and Wan/DiT flow-style backends.

Forbidden:

- Universal adapter.
- All video inpainting models supported.
- EffectErase adapter-ready.
- MiniMax quality-positive before a heldout micro gate passes.
- Final SOTA or top-conference novelty confirmed without final benchmark.

