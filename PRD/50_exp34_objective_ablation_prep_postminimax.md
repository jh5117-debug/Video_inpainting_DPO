# PRD 50: Exp34 Objective Ablation Prep Post-MiniMax

Date: 2026-06-27

## Objective

Prepare Exp27 objective-ablation configs and bugfixes for a future post-MiniMax
study without launching O0-O5, RC-FPO, LocalDPO 24F GPU studies, or any MiniMax
training in this prompt.

## Isolation

- Branch: `research/exp34-objective-ablation-prep-postminimax-20260627`
- Base: `origin/research/exp27-paper-grounded-preference-study`
- Base HEAD: `17b99a421fbf1bb79a446713a1f30ef6c8ecc769`
- HAL worktree: `/home/hj/H20_Video_inpainting_DPO_exp34_objective_prep`
- Scope: CPU/readback/planning only.

Exp34 must not modify Exp30/MiniMax worktrees or outputs, shared trainer code,
or `inference/metrics.py`.

## Exp27 Source State

Completed:

- LocalDPO primitive compatibility and six-video smoke.
- LocalDPO official mask digest path fix.
- true-model SDPO policy/reference forward scan:
  `TRUE_MODEL_PARITY`.
- SDPO true-model 32x4 scan:
  `SDPO_TRUE_MODEL_32X4_SCAN_COMPLETE`.
- SDPO actual tiny-step check:
  `SDPO_TINY_STEP_ACTUAL_CHECK_PASSED`.
- Linear-DPO true-model probe:
  `LINEAR_TRUE_MODEL_PROBE_PASS`.
- Linear-DPO Frozen/EMA 1/10-step:
  `LINEAR_TRUE_MODEL_1_10_STEP_PASSED`.

Pending:

- LocalDPO 24F adaptation.
- O0-O5 controlled objective study.
- RC-FPO.

## Allowed Work

Allowed in Exp34 before MiniMax is released:

- fix LocalDPO `objective_args.manifest` preparation bugs if present;
- prepare O0-O5 config skeletons;
- prepare Linear-DPO Frozen/EMA config skeleton for MiniMax flow target;
- prepare SDPO safe-lambda config skeleton;
- add CPU-only validation tests for config identity and no-run guardrails;
- write reports and registry updates.

Forbidden in this prompt:

- no O0-O5 GPU study;
- no RC-FPO;
- no 50-step LocalDPO;
- no MiniMax training;
- no right-side Exp30 file or output mutation.

## Right-Side Protection

Read-only PAI checks found no active compute process, but Exp30/MiniMax outputs
exist and stale MiniMax locks reserve GPU0 and GPU5. Exp34 does not need a GPU.

## Status

Current status: `EXP34_READBACK_COMPLETED_OBJECTIVE_PREP_PENDING`

Objective-run status: `OBJECTIVE_ABLATION_NOT_RUN_RIGHT_PLUGIN_ACTIVE`

