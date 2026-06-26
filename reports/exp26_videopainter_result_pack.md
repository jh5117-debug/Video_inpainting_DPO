# Exp26 VideoPainter Result Evidence Pack

Status: `EXP26_RESULT_PACK_COMPLETE`

This pack is for group-meeting / paper-discussion evidence. It separates three evidence regimes:

1. Search-dev micro-gate: strong but model-selection evidence.
2. Independent VOR-BG shadow-dev: confirmed Step50 improvement.
3. DAVIS-derived external 49F validation: **not confirmed**, with visible local Step50 artifacts.

The pack does not authorize new training, checkpoint reselection, a universal-adapter claim, or a final SOTA claim.

## Assets

- CSV index: `reports/exp26_videopainter_result_pack.csv`
- Git-friendly local assets: `reports/exp26_videopainter_result_pack_assets/`
- Full search/shadow/external PAI outputs remain under `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/`.

## Case Counts

| Split | Category | Count |
| --- | --- | ---: |
| external-49f | external_difficult_failure | 2 |
| external-49f | external_limited_positive | 3 |
| external-49f | external_step50_artifact | 4 |
| search-dev | search_microgate_example | 6 |
| shadow-dev | clearly_better | 6 |
| shadow-dev | shadow_failure_or_artifact | 2 |
| shadow-dev | slightly_better | 4 |
| shadow-dev | tie | 3 |

## Recommended Slides

- Positive VOR-BG evidence: use 6 shadow-dev `clearly_better` cases showing Step50 removing translucent oval/ring residuals.
- Boundary/texture caution: use 4 `slightly_better` and 3 `tie` shadow-dev cases to show remaining local tradeoffs.
- External caution: use `davis_kid-football`, `davis_motocross-bumps`, `davis_dog-gooses`, `davis_color-run`, and `davis_paragliding` as failure slides. These are the clearest examples where the external split exposes Step50 dark/green local artifacts.
- Limited external positives: `davis_boxing-fisheye`, `davis_bus`, and `davis_hockey` show that Step50 can help some external rows, but not enough for confirmation.

## Scientific Interpretation

The strongest honest phrasing is: LoVI-DPO transfers to VideoPainter on the locked VOR-BG search/shadow distribution and gives cross-backbone adapter evidence for DiffuEraser + VideoPainter. The DAVIS-derived external split shows the adapter is not yet robust across clean-video distribution shifts and should not be called universal.

## Failure Taxonomy

- VOR-BG positives: removal of translucent oval/ring residuals.
- External negatives: dark/green blobs, water/grass/foliage color mismatch, high-motion texture break, thin-structure boundary tint, and local temporal smear.
- No current evidence of unexpected GT leakage, frame-order failure, or first-frame protocol error.
