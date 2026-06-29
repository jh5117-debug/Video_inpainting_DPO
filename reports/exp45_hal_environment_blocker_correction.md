# Exp45 HAL Environment Blocker Correction

Date: 2026-06-29

## Correction Summary

The previous Exp45 response finished quickly because it was executed on
`hal-9000`, not on the intended PAI `dsw-*` host. In that HAL environment,
`/mnt/nas` and `/mnt/workspace` were not mounted, so the real PAI mining roots
were unavailable.

This means the previous Exp45 Milestone C/D/E records are blocker / partial
records only. They are not evidence that targeted pair scale-up mining,
visual relabeling, or formal split construction truly completed.

## 1. Why The Previous Round Ended Quickly

- Host was `hal-9000`.
- Required PAI/NAS roots were missing:
  - `/mnt/nas/hj/H20_Video_inpainting_DPO`
  - `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp44_pai_minimax_targeted_same_source_mining`
  - `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp44_pai_minimax_targeted_same_source_mining/targeted_mining_20260629_161742`
- Because source videos, masks, raw MiniMax candidates, review evidence, and
  output roots were not visible, the session could only record blockers and
  package repo-side manifests. It could not run real PAI mining.

## 2. Which Milestones Are Only Partial / Blocked

- Milestone B handoff filelist:
  `EXP45_H20_FILELIST_PARTIAL_SOURCE_ROOT_UNAVAILABLE`
- Milestone C targeted pair scale-up mining:
  `MINIMAX_TARGETED_SCALEUP_BLOCKED_SOURCE_ROOT_UNAVAILABLE`
- Milestone D strict visual relabel:
  `MINIMAX_TARGETED_RELABEL_BLOCKED_NO_CANDIDATES`
- Milestone E formal Stage2 split:
  `MINIMAX_STAGE2_FORMAL_DATA_PARTIAL`

## 3. Which Milestones Did Not Truly Execute

The following did not truly execute on PAI:

- targeted MiniMax official inference for Exp45 scale-up;
- generation of new Exp45 raw output videos;
- generation of new Exp45 visual review pages;
- strict visual relabeling of new Exp45 candidates;
- construction of an expanded same-source split from new Exp45 candidates;
- dataloader or forward smoke;
- any training, SFT, DPO, or optimizer step.

## 4. New Candidates

New candidates mined in Exp45: `0`.

The previous Exp45 candidate count remained zero because the required PAI/NAS
source root was not available from HAL.

## 5. New Visual Relabel

New visual relabel rows in Exp45: `0`.

No new review pages or videos existed, so no new video evidence was opened or
classified.

## 6. Final Split

The final Exp45-prefixed split is still the copied Exp44 partial split:

- train: `24`
- search: `8`
- shadow: `8`

This does not meet the formal minimum `32/16/16`. Training remains
`TRAINING_NOT_UNLOCKED`.

## 7. What Will Be Done Next On PAI

The next real work must run on a PAI `dsw-*` host where these paths exist:

- `/mnt/nas/hj/H20_Video_inpainting_DPO`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp44_pai_minimax_targeted_same_source_mining`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp44_pai_minimax_targeted_same_source_mining/targeted_mining_20260629_161742`

On PAI, resume:

1. Milestone C: targeted pair scale-up mining;
2. Milestone D: strict visual relabeling of newly mined candidates;
3. Milestone E: formal same-source split construction.

Target remains:

- minimum: `32/16/16`;
- preferred: `64/24/24`.

Still forbidden:

- H20 execution;
- H20 validation;
- SFT/DPO training;
- optimizer steps;
- VOR-Eval use;
- hard comp;
- MiniMax positive / third-adapter-positive language.

## Current Remote Access Note

From HAL, direct SSH to `root@dsw-753014-85f54df947-bkp7h` failed because the
hostname did not resolve. A valid PAI SSH endpoint or alias is required before
remote launch can continue.
