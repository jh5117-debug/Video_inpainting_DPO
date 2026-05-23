# PAI Runbook

## Before Any Run

1. Confirm host and repo:

```bash
set -euo pipefail
date
hostname
whoami
pwd
git rev-parse --show-toplevel
git status --short
```

2. Confirm GPU/Python:

```bash
nvidia-smi
which python
python --version
conda info --envs
```

3. Run the PAI audit and asset preparation helper:

```bash
bash scripts/pai_audit_and_prepare_assets.sh
```

This writes `PRD/pai_audit_pai_node_<timestamp>.md`, creates `current`
symlinks under `data/` and `weights/` for confirmed assets, and writes
`configs/paths/pai.detected.env`. Any unconfirmed root remains commented or
missing in that env file.

4. Load path config:

```bash
source configs/paths/pai.detected.env  # or configs/paths/pai.example.env copied to a private local env
```

5. Check critical env vars:

```bash
for v in VIDEO_DPO_DATA_ROOT YOUTUBE_VOS_ROOT GENERATED_LOSER_ROOT \
  DIFFUERASER_WEIGHT_ROOT PROPAINTER_WEIGHT_ROOT COCOCO_WEIGHT_ROOT \
  MINIMAX_REMOVER_WEIGHT_ROOT OFFICIAL_VIDEODPO_ROOT VC2_WEIGHT_ROOT EXP_OUTPUT_ROOT
do
  printf '%s=%s\n' "$v" "${!v:-}"
done
```

## Four Loser Generation Models

Before Experiment 1/2A/2B data generation, search and verify each model independently:

```bash
find . /home/hj /mnt/workspace /mnt/data \
  -maxdepth 6 \
  \( -iname "*diffueraser*" -o -iname "*propainter*" -o -iname "*cococo*" -o -iname "*minimax*" -o -iname "*remover*" \) \
  2>/dev/null | sort | head -500

grep -RInE "DiffuEraser|diffueraser|ProPainter|propainter|CoCoCo|cococo|MiniMax|minimax|Remover|remover" . 2>/dev/null | head -500 || true
```

Record for each model:

- code path;
- conda/env path;
- weight path;
- generation script;
- README/runbook;
- mask convention: which value means masked/hole and which value means known/keep.

If anything is missing or ambiguous, write `未找到` or `未确认`; do not guess.

## Process Name Convention

All new bash launchers default to:

```bash
export LINGBOT_PROCESS_NAME="${LINGBOT_PROCESS_NAME:-lingbot-phy}"
```

Use `scripts/lingbot_process.sh` for new wrappers. W&B names should remain experiment-specific.

## Completed Official Experiments

Re-check these on PAI before moving/deleting anything:

- VC2 training: `logs/videodpo_vc2_dpo_official_clean/pai-vc2-dpo-official-full-gpu0-3-gb8-step3000-20260521_061414`
- VC2 full VBench: `logs/vbench_full/pai-vc2-official-step3000-full-vbench-20260521_141824`
- DiffuEraser stage1: `logs/official_diffueraser_stage1/pai-official-diffueraser-stage1-full-gpu4-7-gb8-step3000-20260521_072559`
- DiffuEraser stage2: `logs/official_diffueraser_stage2/pai-official-diffueraser-stage2-full-gpu4-7-gb8-step3000-20260521_150540`
- DiffuEraser fullmask VBench: `logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926`

## Smoke Tests Only During Refactor

Do not start full training during structural cleanup. Use:

```bash
python -m tools.offline_loser_generation --help
bash official_videodpo_diffueraser_data_fullmask_loser/scripts/run_generate_losers.sh --help
python - <<'PY'
import importlib
for name in ["official_videodpo_diffueraser", "official_videodpo_vc2"]:
    importlib.import_module(name)
    print("OK", name)
PY
```

## Common Failure Checks

- Missing NAS mount: `/mnt/nas` or `/mnt/workspace` not visible.
- Conda solver issue: `GLIBCXX_3.4.31` missing for libmamba.
- Missing runtime packages in active env: `moviepy`, `av`, `omegaconf`.
- CoCoCo/MiniMax weights not confirmed.
- Stale GPU processes with old `lingbotworld-phy` names from previous runs.
