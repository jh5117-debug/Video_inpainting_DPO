# Exp47 H20 MiniMax Pseudo-Success SFT Failure Forensic Audit

Status: EXP47_MANIFEST_ALIGNMENT_PASS

Branch: `research/exp47-h20-minimax-pseudosuccess-forensic-20260629`
Start HEAD: `94d2531a6782914e91bd4629fb477e154cfba98b`
Base: `origin/research/exp46-h20-minimax-exp45-pseudosuccess-sft-20260629`
Exp46 final HEAD: `94d2531a6782914e91bd4629fb477e154cfba98b`

## Scope

Exp47 is an H20-only forensic audit of the Exp46 pseudo-success SFT30 failure. It is report/script-only plus no-grad or no-optimizer audits. It must not train, run DPO, run GT-only SFT, run 100-step, modify Exp46 outputs, or edit shared trainer/metrics/MiniMax official source.

## Milestones

- A readback: complete (`EXP47_READBACK_READY`)
- B manifest/path/frame alignment audit: complete (`EXP47_MANIFEST_ALIGNMENT_PASS`)
- C pseudo-success teacher quality audit: pending
- D Step30 movement direction audit: pending
- E region loss/mask/weight contribution audit: pending
- F strict pseudo-success relabel proposal: pending
- G final root-cause decision: pending

## Initial Exp46 Failure Summary

Search deltas full/mask/boundary/outside: `-4.612642/-0.548113/-1.591353/-4.812891`.

Shadow deltas full/mask/boundary/outside: `-3.366753/-5.674479/-3.636023/-3.029058`.

Visual worse rows: `48/48`; better rows: `0/48`.


## Milestone B Manifest / Path / Frame Alignment Audit

Status: `EXP47_MANIFEST_ALIGNMENT_PASS`

Rows audited: `112` total (`24` search, `24` shadow, `64` train by Exp46 runner manifests). Failed rows: `0`. Split overlap total: `0`.

Active target field: `winner_path` from `manifests/exp46_runner_pseudosuccess_*`. These active targets are H20-local pseudo-success frame directories extracted from Exp45 pseudo-success mp4 files. Original Exp45 `target_frames_dir` fields may be empty or partial and are preserved only as provenance; they were not the active Exp46 training/eval target.

Checks passed for all rows: active paths exist, active paths are H20-local, no active path points to PAI `/mnt/nas` or HAL, target is pseudo-success rather than GT/condition/failure, target frames match pseudo-success mp4, frame counts and resolutions are consistent, RGB/BGR sanity passes, mask is non-empty with expected polarity, no VOR-Eval rows, no hard-comp rows, and train/search/shadow scene overlap is zero.

Conclusion: Exp46 pseudo-success SFT30 regression is not explained by manifest/path/frame alignment, target identity, mask polarity, VOR-Eval leakage, or hard-comp leakage. The remaining hypotheses are teacher quality/global drift, Step30 movement direction, and region loss contribution/globalization risk.
