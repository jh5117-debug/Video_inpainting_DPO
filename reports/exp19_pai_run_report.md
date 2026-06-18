# Exp19 PAI Run Report

Date: 2026-06-18

## PAI Context

```text
host = dsw-753014-dc85766cb-4v2jj
repo = /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO
sync_strategy = target_import_rsync
```

The PAI main worktree was dirty, so Exp19 was synced by targeted rsync only:

```text
exp19_boundary_gated_flow_adapter_dpo/
experiment_registry/exp19_boundary_gated_flow_adapter_dpo/
PRD/40_exp19_boundary_gated_flow_adapter_dpo.md
reports/exp19_*.md
```

No reset, clean, deletion, or old-result overwrite was performed.

## PAI Precheck

Passed:

```text
python -m py_compile exp19_boundary_gated_flow_adapter_dpo/code/*.py
bash -n exp19_boundary_gated_flow_adapter_dpo/scripts/*.sh
```

GPU check showed GPUs 0-6 effectively idle and GPU 7 partially occupied, but
training was not launched because architecture preflight blocked first.

## Launcher Result

Command:

```bash
bash exp19_boundary_gated_flow_adapter_dpo/scripts/launch_exp19_overnight_pai.sh
```

Result:

```text
EXP19_LAUNCH_EXIT=3
BLOCKED_AT_ARCHITECTURE_PREFLIGHT
```

Log:

```text
logs/pipelines/exp19_boundary_gated_flow_adapter_dpo_overnight.log
```

## Why It Stopped

The launcher ran the Exp19 architecture preflight and stopped before flow-cache
export or training. The shared `UNetMotionModel` residual path is unsafe for the
requested multi-scale adapter:

- down + mid residuals can double-add down residuals;
- down-only residuals use a different legacy T2I-adapter contract;
- mid-only would not match the Exp19 method definition.

No Exp19 checkpoint, dpo_diag, DAVIS metric, or visual result was produced.
