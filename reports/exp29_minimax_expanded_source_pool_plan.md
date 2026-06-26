# Exp29 MiniMax Expanded Source-Pool Plan

Date: 2026-06-26

Status: `MINIMAX_EXPANDED_SOURCE_POOL_PLAN_BLOCKED_INSUFFICIENT_AUDIT_ROWS`

## Scope

This milestone plans a larger MiniMax data-yield gate only. It does not run MiniMax inference, recipe search, 30-step micro, or training.

## Source Audit

- Audit CSV: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp25_vor/reports/vor_triplet_audit64_semantic.csv`
- Audit rows: 64
- Valid aligned rows: 63
- Previously used source32 rows excluded: 32
- Remaining valid rows after exclusion: 31
- Required expanded source count: 96 or 128

## Remaining Inventory

- Manifest: `exp29_or_adapter_feasibility/manifests/minimax_expanded_source_pool_v2.jsonl`
- Manifest SHA256: `bb31cfa5abd320dc88a5471036a3b2bb54b91257d3f65380dc43ecdf29c60929`
- Source type counts: {'BLENDER': 8, 'REAL': 23}
- Mask bucket counts: {'large': 14, 'medium': 12, 'small': 5}

All rows in this manifest are marked `eligible_for_generation=false` because the available inventory is too small for the preregistered expanded first pass.

## Seed Rule

- `seed_a = 20260626`
- `seed_b = 20260627`, conditional only
- `seed_c` disabled

## Decision

`MINIMAX_EXPANDED_GENERATION_BLOCKED`

Do not launch MiniMax expanded candidate generation from this audit64 pool. Need a larger group-disjoint source audit before first-pass generation.

The blocker is source-pool size, not GPU availability. The next minimal step is a larger group-disjoint VOR source audit or a non-VOR OR diagnostic source audit. Do not run recipe search or 30-step micro from this 31-row remainder.
