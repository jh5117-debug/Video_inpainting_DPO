# Exp47 H20 MiniMax Pseudo-Success SFT Failure Forensic Audit

Status: EXP47_STEP30_MOVEMENT_AUDITED

Branch: `research/exp47-h20-minimax-pseudosuccess-forensic-20260629`
Start HEAD: `94d2531a6782914e91bd4629fb477e154cfba98b`
Base: `origin/research/exp46-h20-minimax-exp45-pseudosuccess-sft-20260629`
Exp46 final HEAD: `94d2531a6782914e91bd4629fb477e154cfba98b`

## Scope

Exp47 is an H20-only forensic audit of the Exp46 pseudo-success SFT30 failure. It is report/script-only plus no-grad or no-optimizer audits. It must not train, run DPO, run GT-only SFT, run 100-step, modify Exp46 outputs, or edit shared trainer/metrics/MiniMax official source.

## Milestones

- A readback: complete (`EXP47_READBACK_READY`)
- B manifest/path/frame alignment audit: complete (`EXP47_MANIFEST_ALIGNMENT_PASS`)
- C pseudo-success teacher quality audit: complete (`EXP47_TEACHER_GLOBAL_DRIFT_CONFIRMED`)
- D Step30 movement direction audit: complete (`EXP47_STEP30_MOVEMENT_AUDITED` / `SFT_LOSS_OR_TARGET_PATH_BUG`)
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


## Milestone C Pseudo-Success Teacher Quality Audit

Status: `EXP47_TEACHER_GLOBAL_DRIFT_CONFIRMED`

Rows audited: `48` search/shadow rows. Codex inspected `4` contact review pages covering all search24 and shadow24 pseudo-success teacher rows.

Label counts: strict clean `0`, usable local-only `22`, global drift `26`, outside bad `0`, boundary bad `0`.

Mean sampled pseudo target vs V_bg metrics: full PSNR `32.370202`, mask PSNR `28.258355`, boundary PSNR `25.815867`, outside PSNR `32.921862`, outside L1 `0.017477`. Mean absolute brightness delta `0.014741`, color histogram distance `0.078622`, low-frequency drift proxy `0.014869`. Mask removal PSNR gain over condition is `18.313552`, confirming local signal exists.

Conclusion: pseudo-success targets are not clean global teachers. They contain useful local removal signal, but the teacher set is dominated by global-drift and local-only rows. Exp46's failure is therefore plausibly a bad/global teacher or localization problem, pending Step30 movement and region-loss audits.


## Milestone D Step30 Movement Direction Audit

Status: `EXP47_STEP30_MOVEMENT_AUDITED`
Root signal: `SFT_LOSS_OR_TARGET_PATH_BUG`

Rows audited: `48` search/shadow rows. Step30 closer to pseudo target: `0`. Step30 closer to `V_bg`: `0`. Bad-teacher-learned rows: `0`. Loss/path/objective-suspect rows: `48`.

Mean Step30-to-pseudo full L1 delta vs Step0: `0.005742`. Mean Step30-to-`V_bg` full/mask/outside L1 deltas vs Step0: `0.011993` / `0.017009` / `0.012223`. Full cosine direction train-vs-teacher `0.262656`, train-vs-`V_bg` `-0.140454`, teacher-vs-`V_bg` `0.025774`.

Conclusion: Step30 does not simply learn the bad pseudo teacher. It moves away from both pseudo target and `V_bg` in the sampled frame-space audit. Since Milestone B passed manifest/path/frame alignment, the next forensic focus is region-loss contribution, runner target construction, or MiniMax flow-objective mismatch rather than pure target identity.
