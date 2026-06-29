# Exp45 PAI MiniMax Pair Scale-Up And H20 Handoff Package

Status: `EXP45_H20_FILELIST_PARTIAL_SOURCE_ROOT_UNAVAILABLE`

## Purpose

Exp45 is a PAI-only continuation of the Exp44 MiniMax data route. It corrects
the previous PAI session boundary drift into H20 execution, then attempts to
scale the Exp44 same-source success/failure handoff from partial `24/8/8` to
at least `32/16/16`, preferably `64/24/24`.

This experiment is a data and handoff packaging lane only. It must not run SFT,
DPO, optimizer steps, H20 validation, H20 GPU jobs, or H20-side path mirroring.

## Branch And Roots

- Branch: `research/exp45-pai-minimax-pair-scaleup-20260629`
- Base: `origin/research/exp44-pai-minimax-targeted-same-source-mining-20260629`
- HAL worktree: `/home/hj/H20_Video_inpainting_DPO_exp45_pai_minimax_pair_scaleup`
- Requested PAI source root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp44_pai_minimax_targeted_same_source_mining/targeted_mining_20260629_161742`
- H20 mirror target: to be consumed by a separate H20 session only.

## Hard Boundaries

- H20 touched: `no`
- H20 GPU used: `no`
- H20 output written: `no`
- Training run: `no`
- Optimizer step: `no`
- DPO run: `no`
- GT-only SFT run: `no`
- VOR-Eval mining/training/threshold use: `no`
- hard comp: `no`
- edits to `inference/metrics.py`: `no`
- edits to shared trainer: `no`
- edits to MiniMax official repo source: `no`
- MiniMax third-backbone-positive language: `forbidden`
- universal adapter language: `forbidden`

## 2026-06-29 Milestone A Readback

Status: `EXP45_PAIR_SCALEUP_READBACK_COMPLETED`.

Readback source:

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/58_exp44_pai_minimax_targeted_same_source_mining.md`
- `experiment_registry/exp44_pai_minimax_targeted_same_source_mining/status.md`
- `experiment_registry/exp44_pai_minimax_targeted_same_source_mining/results.tsv`
- `experiment_registry/exp44_pai_minimax_targeted_same_source_mining/metric_summary.md`
- `experiment_registry/exp44_pai_minimax_targeted_same_source_mining/qualitative_summary.md`
- `reports/exp44_targeted_mining_metrics.csv`
- `reports/exp44_targeted_visual_relabel.csv`
- `reports/exp44_same_source_pair_construction.md`
- `reports/exp44_badnoise_v4_states.md`
- `reports/exp44_stage2_dataset_handoff.md`
- `reports/exp44_h20_handoff_instructions.md`

Verified Exp44 state:

- same-source usable pairs: `40`
- split: `24/8/8`
- minimum formal target: `32/16/16`
- preferred target: `64/24/24`
- bad-noise v4: `MINIMAX_BADNOISE_V4_READY`
- usable H-state records: `26`
- local/random gradient-proxy median ratio: `2.280567`
- handoff status: `MINIMAX_STAGE2_DATA_HANDOFF_PARTIAL`
- training status: `TRAINING_NOT_UNLOCKED`

Scope correction:

- The previous PAI session's H20 path validation / attempted H20-side execution
  is recorded as out of scope for this lane.
- Exp45 will perform no further H20 action.
- Exp45 will only produce PAI-side filelists, checksums where source files are
  accessible, manifest scale-up, relabel reports, and H20 mirror instructions.

Current execution environment caveat:

- Host observed by this Codex session: `hal-9000`
- `/mnt/nas` mounted here: `false`
- `/mnt/workspace` mounted here: `false`
- Consequence: Milestone B can enumerate repository manifests, but cannot
  compute file sizes or SHA256 for absolute PAI/NAS artifacts unless those
  roots become available in this session.

Next milestone:

- Build the H20 handoff filelist/checksum package from Exp44 pseudo-success
  manifests if source roots are available.
- If source roots remain unavailable, write a blocked/partial handoff package
  with exact missing roots and no fabricated checksums.

## 2026-06-29 Milestone B H20 Handoff Filelist

Status: `EXP45_H20_FILELIST_PARTIAL_SOURCE_ROOT_UNAVAILABLE`.

Generated PAI-only handoff package:

- `reports/exp45_h20_required_filelist.txt`
- `reports/exp45_h20_required_sha256.txt`
- `reports/exp45_h20_required_filelist.csv`
- `reports/exp45_h20_handoff_package.md`
- `reports/exp45_h20_handoff_package.json`

Filelist construction:

- scanned Exp44 manifest rows: `120`
- required absolute paths found: `262`
- paths visible in this session: `0`
- paths missing in this session: `262`
- total visible file size: `0`
- `/mnt/nas` available: `false`
- `/mnt/workspace` available: `false`

Repository-side checksums were computed for the Exp44 manifests and reports
that are present in git. Absolute PAI/NAS raw-output artifacts were not visible
from this session, so their SHA256 entries are explicitly marked
`missing_in_current_session` / `NA` rather than fabricated.

PAI did not execute any H20 mirror, H20 validation, H20 training, DPO, SFT, or
optimizer step.

## 2026-06-29 Milestone C Targeted Pair Scale-Up Mining

Status: `MINIMAX_TARGETED_SCALEUP_BLOCKED_SOURCE_ROOT_UNAVAILABLE`.

Milestone C did not launch MiniMax inference because the current session cannot
access the PAI/NAS source and output roots:

- `/mnt/nas`: `missing`
- `/mnt/workspace`: `missing`
- requested Exp44 source root: `missing`
- requested Exp45 output root: `missing`

No empty candidate manifest was fabricated. No GPU task, H20 action, training,
DPO, optimizer step, VOR-Eval use, or hard comp occurred.

Reports:

- `reports/exp45_targeted_scaleup_mining.md`
- `reports/exp45_targeted_scaleup_mining.csv`
- `reports/exp45_targeted_scaleup_summary.json`
