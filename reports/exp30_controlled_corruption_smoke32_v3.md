# Exp30 Controlled-Corruption Smoke32 V3

Status: `CONTROLLED_CORRUPTION_SMOKE32_V3_LOW_YIELD`

- Candidate count: 16
- Source count: 16
- All classification counts: `{'TRIVIAL_BAD': 8, 'MEDIUM_HARD_ELIGIBLE': 8}`
- Primary classification counts: `{'TRIVIAL_BAD': 8, 'MEDIUM_HARD_ELIGIBLE': 8}`
- Primary usable count: 8 / required 7
- Primary technical-valid count: 16
- Primary outside-fail count: 0

The v3 controlled fallback follows the preregistered profile schedule from `exp30_controlled_corruption_v3_plan.json`: CC-v3-B for all sources, CC-v3-A on six repair sources, and CC-v3-C on two affected soft sources.  It is still only one component of Smoke16 v3 and does not unlock Gate64 or adapter training by itself.
