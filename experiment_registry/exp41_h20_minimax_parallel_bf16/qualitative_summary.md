# Exp41 Qualitative Summary

No Exp41 MiniMax outputs have been generated yet.

Imported visual state:

- Exp38 R1: `0/13` clear visual better, `9/13` worse/tradeoff, `4/13`
  tie/local numeric gain.
- Exp40 LocalDPO v3 pool: Codex reviewed all 19 temporal-strip review pages
  covering selected `train64/search24/shadow24`; this was a data-pool review
  only.
- Exp40 Step0 baseline: Codex reviewed 42 pages covering all 112 baseline rows.

Exp41 must not promote any PASS/POSITIVE status without raw videos, temporal
strips, crops, per-video metrics, aggregate metrics, and visual review CSV.

## 2026-06-29 Data Audit

No Exp41 qualitative model review has been performed. Evidence availability is
now complete for the audited manifests: raw outputs and review assets resolve
on H20, and the Exp40 raw/source/winner/mask mp4 first-frame decode audit passed.

## 2026-06-29 BF16 Preflight

No qualitative model review was performed. This milestone validates runtime
stability only: no SIGFPE/OOM/CUDA/Xid occurred in P0-P7.

## 2026-06-29 Official Protocol Audit

Codex opened the local contact sheets generated from all `16` pulled midframe
review sheets and all `16` temporal-strip sheets, and decoded all `16`
side-by-side mp4s with cv2.

Visual/protocol counts across the official 12-step run and the diagnostic
6-step probe:

| classification | count |
| --- | ---: |
| PROTOCOL_VALID_BASELINE_OK | 9 |
| PROTOCOL_VALID_QUALITY_TRADEOFF | 4 |
| PROTOCOL_VALID_QUALITY_FAIL | 3 |

No mask-polarity reversal, hidden comp, or winner/GT leakage into raw output was
observed. MiniMax Step0 baseline still shows quality issues on several rows,
including over-erasure/fog-like fills, terrain/shore hallucination, and dark
masked-region artifacts. This is protocol-ready evidence only, not visual
quality-positive evidence.
