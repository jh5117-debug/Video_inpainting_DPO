# Exp45 PAI MiniMax Pair Scale-Up And H20 Handoff Package

Status: `MINIMAX_TARGETED_RELABEL_COMPLETED`

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

Status: `MINIMAX_TARGETED_MINING_COMPLETED`.

Real PAI execution resumed on host `dsw-753014-85f54df947-bkp7h` with the
required `/mnt/nas` paths mounted. This supersedes the earlier HAL blocker
record for Milestone C.

PAI run root:

- `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp45_pai_minimax_pair_scaleup/scaleup_mining_20260629_190229`

Mining summary:

- selected source groups: `6`
- new candidates mined: `72`
- automatic success candidates: `38`
- automatic medium-hard failure candidates: `26`
- auto too-close candidates: `6`
- auto fogging / over-erasure candidates: `2`
- auto overlap groups: `5`
- auto same-source pair capacity from new candidates alone: `16`
- MiniMax inference launched on PAI GPU0/GPU1: `true`
- OOM/CUDA/Xid observed: `false`
- H20 touched: `false`
- training run: `false`
- optimizer step: `false`
- VOR-Eval used: `false`
- hard comp used: `false`

Reports:

- `reports/exp45_targeted_scaleup_mining.md`
- `reports/exp45_targeted_scaleup_mining.csv`
- `reports/exp45_targeted_scaleup_summary.json`
- `reports/exp45_targeted_scaleup_group_yield.csv`

Manifests:

- `exp45_pai_minimax_pair_scaleup/manifests/exp45_targeted_source_manifest.jsonl`
- `exp45_pai_minimax_pair_scaleup/manifests/exp45_targeted_candidates_all.jsonl`
- `exp45_pai_minimax_pair_scaleup/manifests/exp45_targeted_success_auto.jsonl`
- `exp45_pai_minimax_pair_scaleup/manifests/exp45_targeted_failure_auto.jsonl`

## 2026-06-29 Milestone D Strict Visual Relabel

Status: `MINIMAX_TARGETED_RELABEL_COMPLETED`.

Strict relabel was performed after opening all `8` generated review pages
covering the `64` automatic success/failure candidates. The relabel pass used
the real PAI raw outputs and temporal strips; automatic metrics were treated as
guardrails only.

Relabel summary:

- total candidates relabeled: `72`
- review pages inspected: `8`
- accepted `SUCCESS_CLEAN`: `8`
- accepted `SUCCESS_USABLE` including clean: `28`
- accepted `FAILURE_MEDIUM_HARD`: `22`
- rejected `BORDERLINE_REJECT`: `14`
- rejected `FAILURE_FOGGING`: `2`
- rejected `FAILURE_TOO_CLOSE`: `6`
- same-source groups with both accepted success/failure: `4`
- one-to-one same-source pair precheck from new rows: `8`
- capped same-source combination precheck from new rows: `16`
- H20 touched: `false`
- training run: `false`
- optimizer step: `false`
- VOR-Eval used: `false`
- hard comp used: `false`

Reports:

- `reports/exp45_visual_relabel.md`
- `reports/exp45_visual_relabel.csv`
- `reports/exp45_visual_relabel_summary.json`
- `reports/exp45_visual_relabel_group_yield.csv`
- `reports/exp45_visual_review_page_index.csv`

Manifests:

- `exp45_pai_minimax_pair_scaleup/manifests/exp45_success_clean.jsonl`
- `exp45_pai_minimax_pair_scaleup/manifests/exp45_success_usable.jsonl`
- `exp45_pai_minimax_pair_scaleup/manifests/exp45_failure_medium_hard.jsonl`
- `exp45_pai_minimax_pair_scaleup/manifests/exp45_rejected.jsonl`
- `exp45_pai_minimax_pair_scaleup/manifests/exp45_visual_relabel_all.jsonl`

## 2026-06-29 Milestone E Formal Stage2 Handoff

Status: `MINIMAX_STAGE2_FORMAL_DATA_PARTIAL`.

No new Exp45 pairs were available, so the largest valid split remains the
Exp44 partial handoff copied into Exp45-prefixed manifests:

- pseudo-success distillation: `24/8/8`
- GT distillation: `24/8/8`
- same-source preference: `24/8/8`
- formal minimum: `32/16/16`
- preferred target: `64/24/24`
- scene-group overlap: `0`
- training status: `TRAINING_NOT_UNLOCKED`

Reports:

- `reports/exp45_stage2_formal_handoff.md`
- `reports/exp45_stage2_formal_handoff.csv`
- `reports/exp45_stage2_formal_handoff_summary.json`
- `reports/exp45_h20_handoff_instructions.md`

## 2026-06-29 Milestone G Paper Positioning

Status: `MINIMAX_DATA_SIGNAL_EMERGING_PAIR_YIELD_WEAK`.

Exp45 remains partial and does not count MiniMax as third adapter evidence.
DiffuEraser and VideoPainter remain the main positive adapter evidence. The
next minimal experiment is to resume PAI targeted mining from a NAS-mounted
session and reach at least `32/16/16` before H20 training is unlocked.

Report:

- `reports/exp45_minimax_paper_positioning.md`

## 2026-06-29 HAL Environment Correction

Status: `EXP45_HAL_ENVIRONMENT_BLOCKER_CORRECTION_RECORDED`.

The previous Exp45 C/D/E records were produced on `hal-9000`, where `/mnt/nas`
and `/mnt/workspace` were unavailable. They are blocker / partial records only,
not evidence of real PAI targeted mining or visual relabel completion.

Correction report:

- `reports/exp45_hal_environment_blocker_correction.md`

Real C/D/E must be resumed on a PAI `dsw-*` host with the required NAS paths
mounted.
