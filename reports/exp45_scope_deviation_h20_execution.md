# Exp45 Scope Deviation Record: H20 Execution

Date: 2026-06-29

## Summary

The current Exp45 prompt explicitly identifies that the previous PAI session
crossed into H20-side execution while trying to validate Exp44 handoff paths.
That behavior is recorded here as out of scope for a PAI-only lane.

## What Is Out Of Scope

For Exp45, the following actions are forbidden:

- touching the H20 worktree;
- using H20 GPUs;
- writing H20 outputs;
- modifying the H20 Exp43 runner;
- running H20 pseudo-success SFT;
- running GT-only SFT;
- running DPO;
- running any optimizer step.

## Current Correction

Exp45 will perform no further H20-side action. It will only generate PAI-side
artifacts:

- handoff filelists;
- checksums when source files are locally accessible;
- mirror instructions for a separate H20 session;
- scaled same-source mining artifacts;
- strict visual relabel reports;
- Stage2 handoff manifests.

## Practical Consequence

H20 path validation remains a later H20-session responsibility. This PAI
session may write the command template and expected target paths, but it must
not execute the mirror, path validation, dataloader, or training on H20.

## Status

`EXP45_SCOPE_CORRECTION_RECORDED`
