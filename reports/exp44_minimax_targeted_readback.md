# Exp44 MiniMax Targeted Same-Source Mining Readback

Status: `EXP44_TARGETED_READBACK_COMPLETED`

Branch: `research/exp44-pai-minimax-targeted-same-source-mining-20260629`

Base branch: `origin/research/exp42-pai-minimax-successful-removal-badnoise-20260629`

Start HEAD: `efc07b2e9f5e84fe488b433b812a9f0bc72debaf`

This milestone performed readback and source-group planning only. It did not launch GPU mining, SFT, DPO, optimizer steps, or H20 actions.

## Files Read

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/56_exp42_pai_minimax_successful_removal_badnoise.md`
- H20 read-only PRD if available: `/home/hj/H20_Video_inpainting_DPO_exp43_minimax_stage2_sft/PRD/57_exp43_h20_minimax_stage2_sft_runner.md`
- `experiment_registry/exp42_pai_minimax_successful_removal_badnoise/status.md`
- `experiment_registry/exp42_pai_minimax_successful_removal_badnoise/results.tsv`
- `experiment_registry/exp42_pai_minimax_successful_removal_badnoise/metric_summary.md`
- `experiment_registry/exp42_pai_minimax_successful_removal_badnoise/qualitative_summary.md`
- `reports/exp42_pai_minimax_data_readback.md`
- `reports/exp42_minimax_official_successful_removal_mining.md`
- `reports/exp42_minimax_official_successful_removal_mining.csv`
- `reports/exp42_minimax_successful_removal_visual_review.md`
- `reports/exp42_minimax_successful_removal_visual_review.csv`
- `reports/exp42_minimax_successful_removal_summary.json`

H20 Exp43 reports read read-only when present:

- `/home/hj/H20_Video_inpainting_DPO_exp43_minimax_stage2_sft/PRD/57_exp43_h20_minimax_stage2_sft_runner.md`: `True`
- `/home/hj/H20_Video_inpainting_DPO_exp43_minimax_stage2_sft/reports/exp43_h20_stage2_sft_ladder.md`: `True`
- `/home/hj/H20_Video_inpainting_DPO_exp43_minimax_stage2_sft/reports/exp43_h20_stage2_sft_ladder_metrics.csv`: `True`
- `/home/hj/H20_Video_inpainting_DPO_exp43_minimax_stage2_sft/reports/exp43_h20_stage2_sft_ladder_visual_review.csv`: `True`
- `/home/hj/H20_Video_inpainting_DPO_exp43_minimax_stage2_sft/reports/exp43_h20_bf16_safe_preflight.md`: `True`

## Exp42 Source Sets

- Success rows: `52`
- Failure rows: `80`
- Success scene groups: `18`
- Failure scene groups: `29`
- Existing same-source overlap groups: `7`
- Success-only groups: `11`
- Failure-only groups: `22`
- Fallback groups preregistered: `16`

Existing overlap groups:

- `BLENDER_FOREST026`
- `BLENDER_GRASS001`
- `BLENDER_MOUNTAIN002`
- `REAL_ENV059_00001`
- `REAL_ENV068_00002`
- `REAL_ENV097_00001`
- `REAL_ENV105_00001`

Success-only groups that need failure mining:

- `BLENDER_DESERT004`
- `BLENDER_FOREST009`
- `BLENDER_FOREST017`
- `BLENDER_FOREST018`
- `BLENDER_FOREST023`
- `BLENDER_FOREST028`
- `BLENDER_OFFICE002`
- `BLENDER_SCHOOL004`
- `REAL_ENV047_00001`
- `REAL_ENV105_00002`
- `REAL_ENV105_00004`

Failure-only groups that need success mining:

- `BLENDER_FOREST013`
- `BLENDER_FOREST015`
- `BLENDER_FOREST021`
- `BLENDER_FOREST030`
- `BLENDER_FOREST049`
- `BLENDER_GRASS002`
- `BLENDER_MOUNTAIN007`
- `BLENDER_MOUNTAIN011`
- `BLENDER_RIVER004`
- `BLENDER_RIVER012`
- `BLENDER_STREET001`
- `REAL_ENV045_00001`
- `REAL_ENV046_00003`
- `REAL_ENV067_00002`
- `REAL_ENV080_00003`
- `REAL_ENV086_00002`
- `REAL_ENV099_00001`
- `REAL_ENV103_00001`
- `REAL_ENV103_00003`
- `REAL_ENV104_00003`
- `REAL_ENV105_00003`
- `REAL_ENV144_00001`

## Why Exp43 Naive GT SFT Failed

Exp43 H20 reports show the runner itself is viable, including BF16-safe one-batch and true 8GPU 30-step execution, but naive GT SFT was strongly quality-negative. The prompt-provided source-of-truth records search deltas full/mask/boundary/outside `-5.8331` / `-4.6745` / `-4.7009` / `-7.5941` and shadow deltas `-6.5506` / `-4.2232` / `-5.3735` / `-8.4532`, with Ewarp worsening. Therefore Exp44 must not train yet; it must first build same-source MiniMax-native success/failure data.

## What Exp44 Does Differently

Exp44 does not repeat Exp38/Exp40 R1/R2/R3 and does not run direct GT SFT. It targets the exact bottleneck from Exp42: same-source success/failure overlap was only `7`, while bad-noise v3/v4 requires at least `24` usable pairs.

## Preregistered Target Groups and Budget

- Group C existing overlap: `7` groups; add up to 8 candidates initially and 16 max per group to improve density and relabel purity.
- Group A success-only: `11` groups; add up to 12 candidates initially and 24 max to find medium-hard same-source failures.
- Group B failure-only: `22` groups; add up to 12 candidates initially and 24 max to find clean/usable same-source successes.
- Group D fallback: `16` near-miss groups; use only if A/B/C cannot reach the pair gate.

No infinite sampling is allowed. Source group list is locked by `reports/exp44_source_group_plan.csv` and `reports/exp44_source_group_plan.json` before mining.

## Promotion Gates

- Minimum usable same-source success/failure pairs: `24`.
- Target usable pairs: `48`.
- All success/failure labels require strict visual relabeling before pair construction.
- If <24 usable pairs, stop at `MINIMAX_SAME_SOURCE_PAIR_YIELD_INSUFFICIENT`; do not build Stage2 training data or run SFT/DPO.

## PAI GPU Readback

PAI hostname: `dsw-753014-85f54df947-bkp7h`.

Read-only GPU check at `2026-06-29T15:18:00+08:00` found GPU0 and GPU1
occupied by non-Exp44 root-owned jobs:

- GPU0: PID `2174541`, `python3 tools/qxq_sample_valtest_v0.py`, about
  `108856 MiB`, PGID `2109141`.
- GPU1: PID `2172517` was visible in `nvidia-smi pmon` using about
  `65248 MiB`; it exited or changed state before `/proc` cmdline readback.

Other visible compute jobs were also unrelated `qxq_sample_valtest_v0.py`
processes on GPU2/GPU3/GPU5/GPU6. These are not old MiniMax Exp30/35/36/37/38/40/42/44
project processes, so no signal was sent and no cleanup was attempted.

Exp44 did not launch GPU mining in Milestone A. Next GPU work must re-check
GPU0/GPU1 and only proceed if they are truly free or if a user explicitly
authorizes cleanup of those specific jobs.
