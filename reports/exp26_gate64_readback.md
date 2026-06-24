# Exp26 Gate64 Milestone Readback

Status: `GATE64_READBACK_COMPLETED_PROTOCOL_LOCKED`

- train source manifest: `exp26_videopainter_dpo_v2/manifests/vp2_vor_bg_train_source_128.jsonl`
- train source SHA256: `68a862c18ad46271757d96c6b5483c76aeb2cfdfb5cae21a452e215b735daaa6`
- Gate64 manifest: `exp26_videopainter_dpo_v2/manifests/vp2_gate64_source_manifest.jsonl`
- Gate64 manifest SHA256: `b904be82d58ab7cd897c6759b7351e262f61397d9f90d84df05ae42300dbffb6`
- config: `exp26_videopainter_dpo_v2/configs/vp2_mixed_br_mask_v1.json`
- PAI status: `blocked_host_key_changed_ed25519_SHA256_xDOCAS_fw0Bs5m9HizeRi1mkYOcIotlm4CxcfWwpqk`

Files read include PRD/00, PRD/01, PRD/48, registry status, Gate16 final review, Probe4 mask audit, 49F sampler parity, historical BR mask audit, source split statistics, and Exp26 source/mask/generation code.

Already completed: L0-L4, Probe4, Gate16 final video review.
Pending: Gate64 extraction, mask generation, official inference, metrics, dense video review, and only then DPO micro-training.

Banned repeats: no Gate16 rerun, no failed-row replacement, no Gate64 generation before this locked protocol, no DPO training.
