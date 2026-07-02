# Exp59 VOID Kubric Gate8 Inference Readback

Date: 2026-07-02

Branch: `research/exp59-void-kubric-gate8-inference-20260702`

Base: `origin/research/exp58-void-native-data-diagnostic-20260702`

## Status

`EXP59_READBACK_DONE`

Gate8 input status: `EXP59_GATE8_INPUT_WEAK`

## Input Audit

- Manifest exists: `manifests/exp58b_void_native_kubric_gate8.jsonl`
- Rows: 8
- Data root: `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/void/kubric_exp58b/gate8/gate8`
- Path existence: 8/8 rows have `rgb_full`, `rgb_removed`, `quadmask`, and `metadata`
- Decode: 8/8 rows passed video decode on PAI
- Frame count: all decoded as 24 frames
- FPS: all decoded as 8 fps
- Resolution: all decoded as 128x128
- Quadmask values: every row contains `0|63|127|255`
- Regions present: object, overlap, affected, and background regions were recorded in the Exp58B manifest
- VOR mixing: none; all paths are under the Kubric Exp58B native-data root

## Target-Hit Caveat

All 8 metadata rows have `target_hit=false`. In this context, the removed target object was generated and paired, but the renderer metadata did not mark the sampled removal target as a successful target-hit case. This weakens the data for adapter training because the paired counterfactual may not express the intended object-interaction removal cleanly enough.

The data is still sufficient for an official VOID inference diagnostic: it is native Kubric-format paired data with valid rgb_full/rgb_removed/quadmask videos, so it can test whether official VOID inference can consume the generated native data and what kind of Step0/baseline output it produces.

The data is not sufficient as adapter-training evidence yet. Adapter training would require either target-hit-positive native data or a stronger demonstration that the target-hit=false samples still produce meaningful medium-hard same-model loser candidates without transition-region ambiguity.

## Why No One-Step Or 10-Step

Exp59 is only an official inference diagnostic. Previous VOID rescue runs showed mixed/negative one-step behavior on VOR-derived data, and Exp58B only made the native-data test possible. The next scientific question is whether native Kubric inputs produce technically valid and informative official VOID outputs. Preference forward, zero-gap, one-step, and 10-step remain locked until this diagnostic is reviewed.

## Evidence

- Input audit CSV: `reports/exp59_gate8_input_audit.csv`
- Storage audit CSV: `reports/exp59_storage_audit.csv`
- Start-state summary: `reports/exp59_start_state_summary.json`

## Safety

- Training run: no
- Preference forward: no
- Zero-gap: no
- One-step: no
- 10-step: no
- VOID official source modified: no
- `inference/metrics.py` modified: no
- Shared trainer modified: no
