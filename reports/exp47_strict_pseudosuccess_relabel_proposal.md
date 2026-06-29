# Exp47 Strict Pseudo-Success Relabel Proposal

Status: `EXP47_STRICT_PSEUDOSUCCESS_RELABEL_PROPOSED`

Rows considered: `48` audited search/shadow pseudo-success rows. This proposal does not train or run an optimizer step.

## Proposed Counts

- `SUCCESS_LOCAL_ONLY`: `48`

Teacher global-drift rows from Milestone C: `26`.

Strict clean count: `0`. Local-only count: `48`.

Strict 32/16/16 global-SFT split possible: `False`.

## Decision

The strict global-clean pool is empty, so global pseudo-success SFT is not unlocked. The audited rows do preserve local removal signal, so the next viable direction is localized pseudo-success target construction or same-source preference/DPO, not another global SFT run.

Recommendation: `local pseudo-success target construction or same-source DPO; no global SFT`.

Generated manifests:

- `manifests/exp47_success_clean_strict.jsonl`
- `manifests/exp47_success_local_only.jsonl`
- `manifests/exp47_reject_global_drift.jsonl`
- `manifests/exp47_reject_boundary_outside.jsonl`
