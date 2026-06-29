# Exp45 Pair Scale-Up Readback

Date: 2026-06-29

## Git

- Branch: `research/exp45-pai-minimax-pair-scaleup-20260629`
- Start HEAD: `81ad11ac08267fcc5db8bd0ebe9bd41bc9fca620`
- Base lineage: `origin/research/exp44-pai-minimax-targeted-same-source-mining-20260629`
- Worktree: `/home/hj/H20_Video_inpainting_DPO_exp45_pai_minimax_pair_scaleup`
- Initial status: clean

## Files Read

PRD / registry:

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/58_exp44_pai_minimax_targeted_same_source_mining.md`
- `experiment_registry/exp44_pai_minimax_targeted_same_source_mining/status.md`
- `experiment_registry/exp44_pai_minimax_targeted_same_source_mining/results.tsv`
- `experiment_registry/exp44_pai_minimax_targeted_same_source_mining/metric_summary.md`
- `experiment_registry/exp44_pai_minimax_targeted_same_source_mining/qualitative_summary.md`

Reports:

- `reports/exp44_targeted_mining_metrics.csv`
- `reports/exp44_targeted_visual_relabel.csv`
- `reports/exp44_same_source_pair_construction.md`
- `reports/exp44_badnoise_v4_states.md`
- `reports/exp44_stage2_dataset_handoff.md`
- `reports/exp44_h20_handoff_instructions.md`

## Exp44 Source State

Exp44 already produced a useful but partial same-source package:

- targeted candidates: `452`
- technical failures: `0`
- strict visual relabel review pages inspected: `47`
- `SUCCESS_CLEAN`: `33`
- `SUCCESS_USABLE`: `92`
- usable success total: `125`
- `FAILURE_MEDIUM_HARD`: `137`
- rejected / borderline / non-usable: `190`
- same-source groups with both usable success and medium-hard failure: `10`
- same-source usable pairs: `40`
- current split: `24/8/8`
- split scene overlap: `0`

Bad-noise v4 is ready as a data artifact:

- state records: `40`
- usable H-state records: `26`
- minimum H-state gate: `24`
- H1 states: `20`
- H3 states: `20`
- local/random gradient-proxy median ratio: `2.280567`
- outside-risk median: `0.342387`

The Stage2 handoff remains partial:

- GT distillation: `24/8/8`
- pseudo-success distillation: `24/8/8`
- same-source preference: `24/8/8`
- formal minimum: `32/16/16`
- preferred target: `64/24/24`
- status: `MINIMAX_STAGE2_DATA_HANDOFF_PARTIAL`
- training status: `TRAINING_NOT_UNLOCKED`

## Why Exp45 Exists

Exp44 crossed the minimum same-source pair gate for bad-noise construction, but
not the formal Stage2 handoff split size. Exp45 therefore needs more pair
density and more split capacity, especially search and shadow, before H20
should run pseudo-success SFT.

This lane is PAI-only. It will not run H20 validation, H20 training, DPO,
optimizer steps, or GT-only SFT.

## Current Environment Check

Observed from this session:

- host: `hal-9000`
- `/mnt/nas`: unavailable
- `/mnt/workspace`: unavailable
- `/mnt/workspace/hj/nas_hj`: unavailable

This means repository-side readback and manifest bookkeeping are possible, but
direct file-size and SHA256 checks for absolute PAI/NAS artifacts are blocked
until the PAI/NAS mount is visible.

## Exp45 Plan

Milestone B:

- Build a handoff filelist from Exp44 pseudo-success manifests.
- Compute file size and SHA256 only for files accessible in this session.
- If the PAI source root is unavailable, mark the package partial/blocked and
  provide exact H20 rsync instructions without pretending checksums exist.

Milestone C-D:

- Continue targeted same-source pair scale-up mining and strict visual relabel
  only from VOR-Train-derived source groups.
- No VOR-Eval, no hard comp, no training.

Milestone E:

- Build formal Stage2 splits only if enough same-source pairs exist.
- Minimum formal target: `32/16/16`.
- Preferred target: `64/24/24`.

Milestone F:

- Optional PAI-only dataloader/forward smoke, no optimizer step, only if the
  formal split is ready.

Milestone G:

- Summarize paper positioning. MiniMax is not third-adapter evidence until a
  later H20/PAI training run passes heldout/shadow quality gates.

## Status

`EXP45_PAIR_SCALEUP_READBACK_COMPLETED`
