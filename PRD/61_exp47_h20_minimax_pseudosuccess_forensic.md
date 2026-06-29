# Exp47 H20 MiniMax Pseudo-Success SFT Failure Forensic Audit

Status: EXP47_READBACK_READY

Branch: `research/exp47-h20-minimax-pseudosuccess-forensic-20260629`
Start HEAD: `94d2531a6782914e91bd4629fb477e154cfba98b`
Base: `origin/research/exp46-h20-minimax-exp45-pseudosuccess-sft-20260629`
Exp46 final HEAD: `94d2531a6782914e91bd4629fb477e154cfba98b`

## Scope

Exp47 is an H20-only forensic audit of the Exp46 pseudo-success SFT30 failure. It is report/script-only plus no-grad or no-optimizer audits. It must not train, run DPO, run GT-only SFT, run 100-step, modify Exp46 outputs, or edit shared trainer/metrics/MiniMax official source.

## Milestones

- A readback: complete (`EXP47_READBACK_READY`)
- B manifest/path/frame alignment audit: pending
- C pseudo-success teacher quality audit: pending
- D Step30 movement direction audit: pending
- E region loss/mask/weight contribution audit: pending
- F strict pseudo-success relabel proposal: pending
- G final root-cause decision: pending

## Initial Exp46 Failure Summary

Search deltas full/mask/boundary/outside: `-4.612642/-0.548113/-1.591353/-4.812891`.

Shadow deltas full/mask/boundary/outside: `-3.366753/-5.674479/-3.636023/-3.029058`.

Visual worse rows: `48/48`; better rows: `0/48`.
