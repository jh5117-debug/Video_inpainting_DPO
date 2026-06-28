# Exp40 LocalDPO v3 Pool

Status: `MINIMAX_LOCALDPO_V3_POOL_READY_MINIMUM`

Source rules:

- VOR-Train only
- VOR-Eval used: `false`
- hard comp used: `false`
- condition = `V_obj`
- winner = `V_bg`
- loser = locally corrupted `V_bg`
- candidate rows per source <= 3
- outside preservation enforced by exact outside reinjection and strict metric gates
- boundary safety enforced by metric gates

Selected counts: `{'train': 64, 'search': 24, 'shadow': 24}`
Materialized counts: `{'train': 64, 'search': 24, 'shadow': 24}`
Candidate rows: `336`
Classification counts: `{'MEDIUM_HARD_ELIGIBLE': 282, 'TRIVIAL_BAD': 31, 'TOO_CLOSE': 14, 'HARD_BUT_PLAUSIBLE': 9}`
Scene overlap: `{'train_search': 0, 'train_shadow': 0, 'search_shadow': 0}`

Visual review:

- Codex opened all 19 temporal-strip review pages covering the selected
  `train64/search24/shadow24` rows.
- Review status: `REVIEWED_PASS_TEMPORAL_STRIP_POOL_AUDIT`.
- The selected losers are local corruptions of `V_bg`, not hard comp outputs.
- No run-level global collapse, black/purple failure, or systematic far-outside
  damage was observed in the review pages.
- This is a pool-construction review only. It is not a MiniMax model-quality
  pass, and it does not unlock any positive adapter claim by itself.

Limitations:

- The original target was `train96/search32/shadow32`; this run reached the
  pre-registered minimum `train64/search24/shadow24`.
- The run used existing materialized references and an exact-cache root to avoid
  another full tar scan/extraction. The earlier sequential tar extraction attempt
  was stopped before completion and is not used by this selected pool.
- PAI `hj` could not write to the requested experiments output root, so the
  selected pool outputs are stored under the Exp40 NAS log root.
