# Exp39 H20 Required Manifest Path Audit

Status: `H20_REQUIRED_MANIFEST_PATHS_COMPLETE`

This audit treats review sheets, temporal strips, diagnostic comps, and side-by-side videos as optional visual evidence assets. Training/smoke paths such as condition, winner, mask, loser/raw frames, raw mp4, and bad-noise/state paths are required.

| manifest | required paths | required missing | optional paths | optional missing |
| --- | ---: | ---: | ---: | ---: |
| `exp30_vor_or_multimodel_minimax/manifests/vor_or_gate64_train32_v3.jsonl` | 153 | 0 | 87 | 87 |
| `exp30_vor_or_multimodel_minimax/manifests/vor_or_gate64_heldout16_v3.jsonl` | 80 | 0 | 54 | 54 |
| `exp30_vor_or_multimodel_minimax/manifests/vor_or_gate64_selected_primary_v3.jsonl` | 243 | 0 | 145 | 145 |
| `exp30_vor_or_multimodel_minimax/manifests/vor_or_gate64_candidates_all_v3.jsonl` | 1152 | 0 | 512 | 512 |
| `exp30_vor_or_multimodel_minimax/manifests/vor_or_gate64_rejected_v3.jsonl` | 909 | 0 | 367 | 367 |
| `exp37_minimax_localdpo_badnoise_rescue/manifests/localdpo_or_train32.jsonl` | 160 | 0 | 32 | 32 |
| `exp37_minimax_localdpo_badnoise_rescue/manifests/localdpo_or_heldout16.jsonl` | 80 | 0 | 16 | 16 |
| `exp37_minimax_localdpo_badnoise_rescue/manifests/exp37_badnoise_states.jsonl` | 128 | 0 | 0 | 0 |
| `exp38_minimax_full_adapter_breakthrough/manifests/localdpo_v2_train30_filtered.jsonl` | 150 | 0 | 30 | 30 |
| `exp38_minimax_full_adapter_breakthrough/manifests/localdpo_v2_heldout13_filtered.jsonl` | 65 | 0 | 13 | 13 |
| `exp38_minimax_full_adapter_breakthrough/manifests/badnoise_v2_train30_states.jsonl` | 120 | 0 | 0 | 0 |

Decision:

`H20_REQUIRED_MANIFEST_PATHS_COMPLETE`
