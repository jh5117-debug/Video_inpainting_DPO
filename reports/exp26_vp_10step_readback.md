# Exp26 VideoPainter 10-Step Readback

Date: 2026-06-25

## Gate Status

Completed before this milestone:

- `GATE64_DATA_READY`
- `PRIMARY32_FINAL_LOCKED`
- `VP_L0_L1_PASSED`
- `VP_STEP0_BASELINE_LOCKED`
- `TEMPORAL_REVIEW_PASS`

Pending:

- 10 optimizer step VideoPainter DPO micro gate.
- Step1/step10 search-dev generation and metric/visual review.
- Conditional 50-step only if the pre-registered 10-step gate passes.

## Source Inputs

- Primary-32 final manifest:
  `exp26_videopainter_dpo_v2/manifests/vp2_gate64_primary32_final.jsonl`
- Primary-32 SHA256:
  `82f661f2f30a581a213972533817624217eabb97eba7aaeedc00ee2109e4e716`
- Search-dev step0 run:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/searchdev_step0_official_20260625_131957`
- Step0 review status:
  `TEMPORAL_REVIEW_PASS_DENSE_EVIDENCE`

## Training Protocol

- Frames: `49`
- Loser: final hard-composited VideoPainter loser via `final_loser_video_path`
- Reference: frozen official VideoPainter checkpoint
- Policy initialization: official VideoPainter checkpoint
- Optimizer: AdamW
- LR: `1e-4`
- Betas: `(0.9, 0.95)`
- Epsilon: `1e-8`
- Weight decay: `1e-4`
- Scheduler: `constant`
- Warmup: `500`
- Max grad norm: `1.0`
- Max train steps: `10`
- Checkpoints: step1, step5, step10, last_weights

## Banned Actions

- Do not run 50-step unless the 10-step gate passes.
- Do not run 100-step or any long training.
- Do not regenerate Gate64, primary-32, or search-dev.
- Do not modify Exp1-Exp24 or `inference/metrics.py`.

