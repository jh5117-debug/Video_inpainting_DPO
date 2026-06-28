# Exp38 LocalDPO v2 Codex Review

Status: `MINIMAX_LOCALDPO_V2_FILTERED_POOL_READY`

The deterministic LocalDPO v2 corruption run completed on PAI under `localdpo_v2_20260628`. Codex inspected the 48 selected review sheets locally. The pool is cleaner than prior LocalDPO corruption: corruption is local, mask/outside separation is respected, and no global black/purple collapse was observed.

However, 5 selected rows are too strong/trivial-bad, mainly visible as red/boundary bars or overly harsh local corruption. They are preserved in the raw audit but excluded from rescue training/evaluation.

Filtered manifests:

- train: `exp38_minimax_full_adapter_breakthrough/manifests/localdpo_v2_train30_filtered.jsonl` rows=30 sha256=dd371ff2953da1cb60876351af84af3ca30b95418cc80f5d964adc0d59283ca0
- heldout: `exp38_minimax_full_adapter_breakthrough/manifests/localdpo_v2_heldout13_filtered.jsonl` rows=13 sha256=feed05a2c5ca296313a1f82f7b0d6d22ef6b231d4edf6de16321b341f2385490

Raw selected counts:

- train32: 32 rows, rejected 2
- heldout16: 16 rows, rejected 3

Decision: use filtered `train30 + heldout13` for bad-noise v2 and bounded SFT/DPO rescue. Do not promote the unfiltered pool.
