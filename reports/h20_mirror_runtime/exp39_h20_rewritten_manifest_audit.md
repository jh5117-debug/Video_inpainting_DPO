# Exp39 H20 Rewritten Manifest Audit

Status: `H20_MANIFEST_REWRITE_HAS_MISSING_PATHS`

Manifest count: `11`
Total rows: `713`
Total absolute refs before rewrite: `4496`
Missing rewritten paths: `1256`

| manifest | rows | abs refs | missing | rewritten sha256 |
| --- | ---: | ---: | ---: | --- |
| `exp30_vor_or_multimodel_minimax/manifests/vor_or_gate64_train32_v3.jsonl` | 32 | 240 | 87 | `eb8d174981d8ca2ba126f32b6c6b0b2852263ab33e86168acf655f46d8515e75` |
| `exp30_vor_or_multimodel_minimax/manifests/vor_or_gate64_heldout16_v3.jsonl` | 16 | 134 | 54 | `b48249414b5ad34e7a459cd55ef33d8b198404bebc47a9075232260fb763b320` |
| `exp30_vor_or_multimodel_minimax/manifests/vor_or_gate64_selected_primary_v3.jsonl` | 50 | 388 | 145 | `e481b17d2f66c484eee4124e4c18fefc60345742ded6e5e75a80db2a4576bcb6` |
| `exp30_vor_or_multimodel_minimax/manifests/vor_or_gate64_candidates_all_v3.jsonl` | 256 | 1664 | 512 | `cabd671eb9e17e3d80c733ffbb09ae8c4960ca191d16e89df52a5c74dd1d624e` |
| `exp30_vor_or_multimodel_minimax/manifests/vor_or_gate64_rejected_v3.jsonl` | 206 | 1276 | 367 | `a68de4f8b4119efc094d9c7761fb2783fe866b122e9932507ce38e996e0e94a5` |
| `exp37_minimax_localdpo_badnoise_rescue/manifests/localdpo_or_train32.jsonl` | 32 | 192 | 32 | `32b27c4a2128c3ee97d6795151f21e033a59cfc05e656b684e6de17b8e54a0d8` |
| `exp37_minimax_localdpo_badnoise_rescue/manifests/localdpo_or_heldout16.jsonl` | 16 | 96 | 16 | `654c6e5f6c543cc519780f02f2cb6b34cab1d9ee540945501933d96d15920d2e` |
| `exp37_minimax_localdpo_badnoise_rescue/manifests/exp37_badnoise_states.jsonl` | 32 | 128 | 0 | `304b668db6e9f79a9e3206562392a09342129fc6533896bce31e42573ac2989d` |
| `exp38_minimax_full_adapter_breakthrough/manifests/localdpo_v2_train30_filtered.jsonl` | 30 | 180 | 30 | `33c034a7201a50b8f395f3611fbb811534739ea556ad5226aba6bd2d546bd862` |
| `exp38_minimax_full_adapter_breakthrough/manifests/localdpo_v2_heldout13_filtered.jsonl` | 13 | 78 | 13 | `3cd6734b9f42d9bc448c853d44815812f2bfaaf8c09edac69efbb545621149b3` |
| `exp38_minimax_full_adapter_breakthrough/manifests/badnoise_v2_train30_states.jsonl` | 30 | 120 | 0 | `59805d89e1e52d2df9fb2c5822b9bbc8338092151b5ea46b3e9c22e1e475042a` |

Original manifests are under:
`/home/nvme01/H20_Video_inpainting_DPO/data/h20_mirror/minimax/manifests/pai_original`

H20 rewritten manifests are under:
`/home/nvme01/H20_Video_inpainting_DPO/data/h20_mirror/minimax/manifests/h20_rewritten`

Decision:

`H20_MANIFEST_REWRITE_HAS_MISSING_PATHS`
