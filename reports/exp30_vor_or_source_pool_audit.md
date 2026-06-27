# Exp30 VOR-OR Source Pool Audit

Status: `VOR_OR_SOURCE_POOL_BLOCKED`

## Counts

- Discovered triplets: 192
- Candidate scene groups after exclusions: 80
- Source rows: 80
- Reserve rows: 0
- Rejected rows: 112

## Identity

- Source manifest SHA256: `58696bc504e79eec1342f00cbbb93d244b96d8311f128cf14156c3c6283cb595`
- Reserve manifest SHA256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

## Balance

- Source type counts: `{'REAL': 71, 'BLENDER': 9}`
- Mask bucket counts: `{'medium': 40, 'large': 18, 'small': 22}`

## Preview

- Batch preview pages: 10
- Each source preview page row contains condition, winner, and mask strips.

## Visual Review

- Batch preview pages opened by Codex: 10/10.
- Preview coverage: 80/80 source rows.
- Visual source sanity: condition/winner/mask strips are generally aligned and
  masks are non-empty.
- Observed limitation: the usable cache remainder is heavily REAL-skewed and
  too small for the requested pool128 plus reserve128 design.

## Decision

This milestone uses existing exact extraction caches and does not rescan VOR tar
archives. After excluding previous MiniMax/EffectErase scenes, only 80 usable
scene groups remain, reserve rows are 0, and source-type balance is poor
(`REAL=71`, `BLENDER=9`). The source pool therefore cannot be promoted to
`VOR_OR_SOURCE_POOL_READY`.

Final decision: `VOR_OR_SOURCE_POOL_BLOCKED`.
