# Exp49 ROSE Feasibility Outcome

## Scope

This is a PAI-only feasibility outcome for ROSE as a possible third video-inpainting adapter candidate. It uses official ROSE inference on VOR-Train rows and an isolated code audit. It does not touch H20, does not use VOR-Eval, does not train, and does not run an optimizer step.

## Gate Results

- Asset/env status: `ROSE_ENV_READY` with Python 3.12 venv `/home/hj/venvs/rose_exp49_py312`.
- Official inference smoke: `ROSE_INFERENCE_SMOKE_PASS`, VOR-Train smoke6 technical-valid `6/6`.
- VOR-OR Gate16: `ROSE_VOR_OR_GATE16_PASS`, technical-valid `16/16`.
- Gate16 visual labels after Codex inspection: `ROSE_OUTPUT_USABLE=9`, `MEDIUM_HARD_ELIGIBLE=5`, `SIDE_EFFECT_LEFT=2`, `TRIVIAL_BAD=0`.
- Useful baseline or loser-eligible rows: `14/16`.
- Systematic outside collapse/drift: `0`.

## Adapter Feasibility

ROSE remains `ROSE_TRAINING_FORWARD_BLOCKED`. The official released code exposes a differentiable `WanTransformer3DModel.forward()` and LoRA utility helpers, but no executable official training/finetune script, optimizer/backward loop, explicit loss, or explicit FlowMatch target construction was found. Therefore no one-step or 10-step adapter gate is unlocked.

## Decision

- ROSE can be described as `ROSE_BASELINE_READY` for official-inference baseline evidence on VOR-Train.
- ROSE can be described as `ROSE_LOSER_GENERATOR_USEFUL` because Gate16 produced medium-hard/side-effect failure candidates without systematic collapse.
- ROSE must not be described as adapter-positive or third-backbone evidence.
- Next technical step is an isolated wrapper/target reconstruction design and zero-gap/one-step proof, or use ROSE as a baseline/loser generator while keeping DiffuEraser + VideoPainter as the positive adapter evidence.

## Evidence

- Gate16 report: `reports/exp49_rose_vor_or_gate16.md`
- Gate16 metrics: `reports/exp49_rose_vor_or_gate16_metrics.csv`
- Gate16 visual review: `reports/exp49_rose_vor_or_gate16_visual_review.csv`
- Gate16 summary: `reports/exp49_rose_vor_or_gate16_summary.json`
- Gate16 manifest: `manifests/exp49_rose_vor_or_gate16_manifest.jsonl`
- Gate16 output dir: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp49_pai_rose_adapter_feasibility/vor_or_gate16_20260630_085042`
- Training-forward audit: `reports/exp49_rose_code_adapter_feasibility_audit.md`
