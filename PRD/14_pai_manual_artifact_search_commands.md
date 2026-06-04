# PAI Manual Artifact Search Commands

Updated: 2026-06-04

The H20 audit found several artifact gaps that likely require PAI-side evidence.
I cannot execute commands on PAI. The user should paste the following block into
the PAI terminal and return the three generated text files.

Known PAI result hint from user:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/logs
```

## Copy-paste PAI command block

```bash
cd /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO
mkdir -p /mnt/workspace/hj/experiment_artifact_audit

find /mnt/workspace/hj /mnt/workspace -maxdepth 7 -type d \
  | grep -Ei 'exp4|exp5|newexp5|new_exp5|newexp6|new_exp6|exp7|exp8|exp9|dpo[-_]?diag' \
  | sort > /mnt/workspace/hj/experiment_artifact_audit/pai_exp_dirs.txt

find /mnt/workspace/hj /mnt/workspace -type f \
  | grep -Ei 'diagnostics|all_diagnostics|dpo[-_]?diag|metrics|report|selected.*jsonl|eval|vbench|checkpoint' \
  | sort > /mnt/workspace/hj/experiment_artifact_audit/pai_diag_files.txt

du -sh $(cat /mnt/workspace/hj/experiment_artifact_audit/pai_exp_dirs.txt | head -200) 2>/dev/null \
  > /mnt/workspace/hj/experiment_artifact_audit/pai_exp_dir_sizes.txt || true

echo "===== PAI artifact search summary ====="
wc -l /mnt/workspace/hj/experiment_artifact_audit/pai_exp_dirs.txt \
      /mnt/workspace/hj/experiment_artifact_audit/pai_diag_files.txt \
      /mnt/workspace/hj/experiment_artifact_audit/pai_exp_dir_sizes.txt

echo
echo "Please return these files:"
echo "/mnt/workspace/hj/experiment_artifact_audit/pai_exp_dirs.txt"
echo "/mnt/workspace/hj/experiment_artifact_audit/pai_diag_files.txt"
echo "/mnt/workspace/hj/experiment_artifact_audit/pai_exp_dir_sizes.txt"
```

## What these files will complete

After the user returns the three files, update:

- `PRD/12_experiment_artifact_registry.md`
- `PRD/13_dpo_diag_audit.md`
- `PRD/15_experiment_registry_repair_final_report.md`

Priority PAI evidence to recover:

1. Old Exp5 full run folder and dpo-diagnostics CSV.
2. New Exp5 full run folder and dpo-diagnostics CSV.
3. Exp7 gate training dpo-diagnostics CSV and partial-mask eval report.
4. Clean PAI Exp9 comp dpo-diagnostics CSV and target eval outputs.
5. Exp8 region-loss status, if it was actually launched.

