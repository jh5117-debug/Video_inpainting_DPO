# Exp44 PAI MiniMax Targeted Same-Source Mining

Status: `EXP44_TARGETED_READBACK_COMPLETED`

## Purpose

Exp44 addresses MiniMax's current data bottleneck: Exp42 found row-level successful-removal and failure signals, but only `7` same-source success/failure overlap groups. Exp44 performs targeted second-pass mining and visual relabeling to construct clean same-source pairs for bad-noise v4 and Stage2-style H20 handoff.

## Branch and Roots

- Branch: `research/exp44-pai-minimax-targeted-same-source-mining-20260629`
- Base: `origin/research/exp42-pai-minimax-successful-removal-badnoise-20260629`
- HAL worktree: `/home/hj/H20_Video_inpainting_DPO_exp44_pai_minimax_targeted`
- Requested PAI code root: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp44_pai_minimax_targeted`
- Output root: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp44_pai_minimax_targeted_same_source_mining`
- Log root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp44_pai_minimax_targeted_same_source_mining`
- Runtime root: `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp44_pai_minimax_targeted_same_source_mining`

## Boundaries

- No H20 worktree/output/GPU modification.
- No training by default; if data gates pass, only dataloader / one-batch forward smoke is allowed.
- No optimizer step in this prompt.
- No VOR-Eval training/selection/tuning.
- No hard comp.
- No modifications to `inference/metrics.py`, shared trainers, MiniMax official repo source, or Exp1-Exp43 historical outputs.
- No MiniMax third-backbone-positive, universal-adapter, final-SOTA, or top-conference novelty claims.

## 2026-06-29 Readback and Source Plan

Status: `EXP44_TARGETED_READBACK_COMPLETED`.

Exp42 verified inputs:

- success rows: `52`
- failure rows: `80`
- success scene groups: `18`
- failure scene groups: `29`
- same-source overlap groups: `7`

Plan:

- Group C overlap groups: `7`
- Group A success-only groups: `11`
- Group B failure-only groups: `22`
- Group D fallback groups: `16`

Reports:

- `reports/exp44_minimax_targeted_readback.md`
- `reports/exp44_source_group_plan.csv`
- `reports/exp44_source_group_plan.json`

Next status target: `MINIMAX_TARGETED_MINING_COMPLETED`.

PAI GPU readback: GPU0/GPU1 were occupied by unrelated `qxq_sample_valtest_v0.py`
root-owned jobs during Milestone A readback. They were not old MiniMax project
processes, so Exp44 sent no signals and launched no GPU task.

## 2026-06-29 Targeted Source Manifest and Runner Preparation

Status: `EXP44_TARGETED_SOURCE_MANIFEST_READY`.

Milestone B preparation added Exp44-isolated helper scripts and built the locked
targeted source/seed manifest:

- source rows: `40` from A/B/C groups only;
- fallback groups included: `false`;
- initial candidate budget: `452`;
- missing source rows: `0`;
- manifest SHA256: `5147839e1e2d60e0ecc9c77a438a934918605b5fa550fa58d1e3291df7be168b`.

New helpers:

- `exp44_pai_minimax_targeted_same_source_mining/scripts/build_targeted_mining_manifest.py`
- `exp44_pai_minimax_targeted_same_source_mining/scripts/mine_targeted_candidates.py`

The runner reuses the already audited MiniMax official pipeline and Exp30
`run_pipeline` helper. It preserves the Exp44 protocol: raw output primary,
UniPCMultistepScheduler, `num_inference_steps=12`, `iterations=6`, float16,
no VOR-Eval, no hard comp, no training, and no optimizer step.

Reports/manifests:

- `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_targeted_source_manifest.jsonl`
- `reports/exp44_targeted_source_manifest_summary.json`

GPU mining is still pending actual GPU0/GPU1 availability.
