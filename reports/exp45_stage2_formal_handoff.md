# Exp45 Stage2 Formal Handoff

Status: `MINIMAX_STAGE2_FORMAL_DATA_PARTIAL`

## Result

Exp45 could not scale the data because targeted mining was blocked by missing PAI/NAS roots. The largest valid handoff remains the Exp44 partial split, copied to Exp45-prefixed manifests for clarity.

## Counts

- train/search/shadow: `24/8/8`
- formal minimum: `32/16/16`
- preferred target: `64/24/24`
- scene overlap: `0` for the copied split
- training unlocked: `false`

## Manifests

- `exp45_pai_minimax_pair_scaleup/manifests/exp45_stage2_pseudosuccess_train.jsonl`: 24 rows, sha256 `e4a9d8dd7b039ee5c3296f6b78811a70202207851ec8a384a9c5fbc9bba69a21`
- `exp45_pai_minimax_pair_scaleup/manifests/exp45_stage2_pseudosuccess_search.jsonl`: 8 rows, sha256 `7fb0ab58b13928379e3d7ecd661b457a57395fa4df04b870fea7d405378715d7`
- `exp45_pai_minimax_pair_scaleup/manifests/exp45_stage2_pseudosuccess_shadow.jsonl`: 8 rows, sha256 `2b6371d028fcd960148208b95357ca644f1fe5c2a6b7d0021fe97f15079c0662`
- `exp45_pai_minimax_pair_scaleup/manifests/exp45_stage2_gt_distill_train.jsonl`: 24 rows, sha256 `24a302b62a72478f4db691559047a6f481a47ce626144c27fd1d51ecb0509e12`
- `exp45_pai_minimax_pair_scaleup/manifests/exp45_stage2_gt_distill_search.jsonl`: 8 rows, sha256 `4d47e4b3a4177188d24d6a04c117e5b84800a8b8565474e18181f200599dcbc0`
- `exp45_pai_minimax_pair_scaleup/manifests/exp45_stage2_gt_distill_shadow.jsonl`: 8 rows, sha256 `9aaabdaccb02225edfa138a685a1556771b54150f51512566f0e6298c145bdf0`
- `exp45_pai_minimax_pair_scaleup/manifests/exp45_stage2_preference_train.jsonl`: 24 rows, sha256 `869bb2ab747ca013c2327957ea26736160d010f3534e1e4be9d3952d0551e374`
- `exp45_pai_minimax_pair_scaleup/manifests/exp45_stage2_preference_search.jsonl`: 8 rows, sha256 `3e155c098832100a678afd33772e4e956cb663648561eb2bc6f864373c99d155`
- `exp45_pai_minimax_pair_scaleup/manifests/exp45_stage2_preference_shadow.jsonl`: 8 rows, sha256 `68f8543cbd67ee0fc043a7f7aa5d14a1bf0c7694251381e7c98be499bf2ef990`

## Scope

- H20 touched: `false`
- training run: `false`
- optimizer step: `false`
- VOR-Eval used: `false`
- hard comp used: `false`

## Interpretation

This package is useful as an indexed partial handoff, but it is not formal-data-ready and must not unlock H20 training. PAI should resume targeted mining in a NAS-mounted session to reach at least `32/16/16`.
