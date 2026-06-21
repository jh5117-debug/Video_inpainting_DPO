# Exp23 Git Sync After Pair001

Date: 2026-06-21

## HAL

```text
hostname = hal-9000
worktree = /home/hj/H20_Video_inpainting_DPO_exp23_pool_sweep
branch = research/exp23-two-stage-pool-morphology-sweep
HEAD = 34844d75aba585542b311098417f67c7274f6434
origin branch HEAD = 34844d75aba585542b311098417f67c7274f6434
```

HAL status at audit time contained only the newly written audit log:

```text
?? reports/exp23_hal_safety_after_pair001_eval_audit.txt
```

## PAI

```text
hostname = dsw-753014-dc85766cb-4v2jj
worktree = /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp23_pool_sweep
branch = research/exp23-two-stage-pool-morphology-sweep
HEAD = 34844d75aba585542b311098417f67c7274f6434
origin branch HEAD = 34844d75aba585542b311098417f67c7274f6434
```

PAI status at audit time contained only Exp23 runtime/untracked audit files:

```text
?? exp23_two_stage_pool_morphology_sweep/queue_state.json
?? exp23_two_stage_pool_morphology_sweep/runtime/
?? reports/exp23_pai_safety_after_pair001_eval_audit.txt
```

No Exp23/Phy training process was running at the start of this audit.

## GPU State

PAI GPU5 and GPU6 were idle-level. GPU2 and GPU4 were occupied by external
`python3` compute jobs at the time of audit. GPU7 still had the known ghost
allocation and was not considered for Exp23.

This report is a sync/safety snapshot only.

