# Exp30 Multi-Model OR Smoke16 V3 Aggregate

Status: `MULTIMODEL_OR_SMOKE16_V3_PASS`

- Candidate rows: 64
- Source rows: 16
- Technical-valid candidates: 64/64
- Best-per-source usable: 13/16
- Best-per-source counts: `{'TRIVIAL_BAD': 3, 'MEDIUM_HARD_ELIGIBLE': 13}`
- Model usable counts: controlled v3 primary 13, MiniMax v2 reused 4, ProPainter 2, DiffuEraser 0.

Codex opened all controlled v3 evidence pages and the 8 verified-generator overview pages. MiniMax v2 evidence was reviewed in the prior v2 failure-analysis milestone and is reused without new seed selection. The aggregate pass is driven by controlled-corruption v3; ProPainter and DiffuEraser are technically valid but low-yield on this fixed Smoke16 set.

This unlocks Smoke32 as the next validation step only. Gate64, MiniMax adapter gates, DiffuEraser micro, RC-FPO, and long training remain stopped.
