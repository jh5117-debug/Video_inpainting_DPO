# Exp19 Final Report

Date: 2026-06-18

Status:

```text
BLOCKED_AT_ARCHITECTURE_PREFLIGHT
```

The PAI launcher did not export full flow cache, train, or evaluate because the
requested multi-scale flow-adapter injection is unsafe through the shared
UNetMotionModel residual interface. See:

```text
reports/exp19_preflight_report.md
```

## PAI Run

```text
host = dsw-753014-dc85766cb-4v2jj
repo = /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO
launcher = exp19_boundary_gated_flow_adapter_dpo/scripts/launch_exp19_overnight_pai.sh
exit = 3
```

PAI GPU check showed GPUs 0-6 idle and GPU 7 partially occupied, but no GPU
training was started.

## Implemented Artifacts

```text
exp19_boundary_gated_flow_adapter_dpo/
experiment_registry/exp19_boundary_gated_flow_adapter_dpo/
PRD/40_exp19_boundary_gated_flow_adapter_dpo.md
reports/exp19_context_architecture_audit.md
reports/exp19_injection_point_audit.md
reports/exp19_pai_run_report.md
```

Implemented but not trained:

- ProPainter completed-flow exporter.
- Forward-backward flow confidence helper.
- Flow manifest dataset extension.
- Zero-initialized residual adapter builder.
- Architecture preflight guard.
- PAI launcher guard.

## Blocker

The shared `UNetMotionModel.forward` has an unsafe residual interface for the
requested multi-scale adapter:

1. Passing both `down_block_additional_residuals` and
   `mid_block_additional_residual` activates a ControlNet-style path.
2. The same down residuals are then added again by a second unconditional branch.
3. Passing only down residuals falls into a legacy T2I-adapter path with a
   different shape contract.
4. Running only a mid-block adapter would no longer be Exp19.

The existing DAVIS eval wrapper also cannot load external flow-adapter weights
or feed flow tensors into DiffuEraser, so a matching Exp19 inference wrapper is
required before any metrics can be trusted.

## Decision

No Exp19a/b/c 500-step gates were launched. No checkpoint, dpo_diag, DAVIS10
metric, DAVIS50 metric, or visual case exists.

Current best remains:

```text
Exp11 outer b0.75 S2
```
