# Exp60B Next Steps After Download Blocker

Status: `EXP60B_NEEDS_SOURCE_URL_RESOLUTION_OR_REPLAN`

The safest next step is not D3 mask generation. The data root is incomplete.

Recommended options, in order:

1. Resolve the 11 Pexels source URL failures without changing row identity,
   for example by obtaining valid alternate official Pexels download links for
   the same source video IDs, then rerun sha256/decode verification.
2. If exact source recovery is impossible, open a new preregistered replan that
   replaces only the blocked 11 rows from the eligible Pexels pool, preserving
   train/test disjointness and documenting the replacement seed and reason.
3. Only after 1,100/1,100 files verify on PAI/NAS, generate PAI manifests and
   proceed to D3 mask generation.

Do not start mask generation, loser generation, inference, or DPO from the
partial 1,089-row set.
