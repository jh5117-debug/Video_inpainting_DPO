# Experiment Registry Repair Final Report

Updated: 2026-06-04

Scope: artifact audit, PRD repair, PPT naming repair, and dpo-diag audit only.
No training, DPO run, data generation, checkpoint modification, or model-code
change was performed.

## 1. H20 Experiment Folders Found

H20 SSH path used for audit:

```text
/home/nvme01/H20_Video_inpainting_DPO
```

Scan outputs saved locally under:

```text
reports/experiment_artifact_audit/
```

Key H20 findings:

- New Exp6 stage1/stage2 diagnostics and checkpoints:
  - `experiments/dpo/stage1/20260601_004753_exp6_d2_nocomp_k4_wingap_lose025_beta10_s1s2_4000_stage1`
  - `experiments/dpo/stage2/20260601_004753_exp6_d2_nocomp_k4_wingap_lose025_beta10_s1s2_4000_stage2`
- Exp9 H20 nocomp:
  - `experiments/dpo/stage1/20260603_131758_exp9_youtubevos_d3_nocomp_partialmask_wingap_lose025_stage1_gate1500_h20_stage1`
  - `logs/target_eval/exp9_d3_nocomp_gate_h20_20260604_023243`
- Exp9 H20 comp no-lose:
  - `experiments/dpo/stage1/20260604_080411_exp9_youtubevos_d3_comp_wingap_nolose_stage1_gate1000_h20_stage1`
  - `reports/exp9_nolose_gate_h20_report.md`

Local qualitative folders found under `/home/hj/dpo-2-1-exp`:

- `exp4-data` with 4 PNG files and 0 MP4 files.
- `exp5`
- `new-exp5`
- `new-exp6`
- `exp7-gate1500`
- `Exp7_DPO_Stage1_last`
- `Exp7_DPO_S1_DPO_S2_last`
- `Hybrid_DPO_S1_last__Official_DiffuEraser_base_Stage2`
- `exp9_d3_comp_gate_pai_ckpt500`
- `exp9_d3_nocomp_gate`

## 2. Missing Or Incomplete Artifacts

Current H20 scan did not locate complete run folders for:

- Exp4 full-mask generated-loser training/eval.
- Old Exp5 complete run folder.
- New Exp5 complete run folder.
- Exp7 complete training dpo-diag folder.
- Exp8 completed run folder.
- PAI clean Exp9 comp training/eval folder.

These may be on PAI, especially because user noted PAI artifacts under:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/logs
```

## 3. dpo-diag Found

Found on H20:

- New Exp6:
  - `experiments/dpo/stage1/20260601_004753_exp6_d2_nocomp_k4_wingap_lose025_beta10_s1s2_4000_stage1/dpo_diagnostics.csv`
  - `experiments/dpo/stage2/20260601_004753_exp6_d2_nocomp_k4_wingap_lose025_beta10_s1s2_4000_stage2/dpo_diagnostics.csv`
- Exp9 H20 nocomp:
  - `experiments/dpo/stage1/20260603_131758_exp9_youtubevos_d3_nocomp_partialmask_wingap_lose025_stage1_gate1500_h20_stage1/dpo_diagnostics.csv`
- Exp9 H20 comp no-lose:
  - `experiments/dpo/stage1/20260604_080411_exp9_youtubevos_d3_comp_wingap_nolose_stage1_gate1000_h20_stage1/dpo_diagnostics.csv`

## 4. dpo-diag Missing Or Unconfirmed

- Exp4: not found.
- Old Exp5: not found in current H20 scan; PRD records interpretation but CSV
  must be recovered if available.
- New Exp5: not found in current H20 scan.
- Exp7: not found in current H20 scan.
- Exp8: not found.
- PAI clean Exp9 comp: not available from H20 scan; requires PAI manual search.

## 5. PPT Repairs

New PPT generated:

```text
PRD/PPT_exp_registry_dpo_diag_fixed.pptx
```

Changes:

- Added corrected naming slide.
- Split Old Exp5 and New Exp5.
- Renamed the no-comp rerun as **New Exp6**, not ordinary Exp6.
- Added experiment registry / folder tracking slide.
- Added mandatory dpo-diag slide.
- Added H20/PAI artifact search slide.
- Added corrected experiment narrative chain.
- Avoided traceback, SSH details, and code-error slides.

## 6. New Exp5 / New Exp6 Correction

Correct presentation:

- Old Exp5: unanchored DPO collapse; ranking looked good, visual quality broke.
- New Exp5: comp data-only rerun with winner anchoring; more stable but not
  final.
- New Exp6: no-comp + changed loss diagnostic; reuses Exp5 raw losers and
  switches manifest, not a generic Exp6.

## 7. User Must Return PAI Files

Use:

```text
PRD/14_pai_manual_artifact_search_commands.md
```

Return:

```text
/mnt/workspace/hj/experiment_artifact_audit/pai_exp_dirs.txt
/mnt/workspace/hj/experiment_artifact_audit/pai_diag_files.txt
/mnt/workspace/hj/experiment_artifact_audit/pai_exp_dir_sizes.txt
```

These are needed to complete:

- Old Exp5 registry.
- New Exp5 registry.
- Exp7 registry.
- Exp8 status.
- PAI clean Exp9 comp registry.

## 8. Whether dpo-diag Must Be Rebuilt

Do not rebuild dpo-diag by rerunning training.

If CSV exists on PAI, recover it. If it does not exist:

- mark `diag gap`;
- only summarize qualitative/metric evidence;
- do not present that experiment as complete.

## 9. Can We Continue Later Experiments?

Not until the artifact registry is repaired enough for the current weekly
claims.

Safe next action:

- user runs PAI artifact search;
- update registry from returned PAI files;
- then decide whether Exp8/no-lose/target-domain SFT warmup remains necessary.

Unsafe next action:

- starting new training to compensate for missing documentation;
- presenting New Exp6 as ordinary Exp6;
- presenting any DPO result without dpo-diag.

