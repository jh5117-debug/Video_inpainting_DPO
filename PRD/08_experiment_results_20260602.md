# Experiment Results 2026-06-02

Updated: 2026-06-04 with artifact registry repair.

## Naming Repair

Do not use a generic "Exp6" label in weekly presentation material.

Use:

- **Old Exp5**: collapsed unanchored comp data-only runs.
- **New Exp5**: winner-anchored comp data-only rerun.
- **New Exp6**: winner-anchored no-comp diagnostic rerun.
- **Exp7**: partial-mask task-alignment gate.
- **Exp9**: D3 target-domain partial-mask gate.

## Result Summary With Artifact Status

| Experiment | Result claim | Artifact status |
| --- | --- | --- |
| Exp4 full-mask generated loser | Negative data-quality smoke: full-mask generated loser quality was too poor. | Incomplete. Local `/home/hj/dpo-2-1-exp/exp4-data` contains PNG evidence only; no H20 dpo-diag found. |
| Old Exp5 | Collapsed. Ranking objective can be satisfied while visual quality breaks. | Incomplete on current H20 scan. PRD records diagnostic interpretation; PAI search needed for original folder/CSV. |
| New Exp5 | Winner anchoring improved stability relative to Old Exp5, but not a final stable path. | Qualitative folder exists locally; H20 dpo-diag not found. PAI search needed. |
| New Exp6 | No-comp diagnostic. Some long-prompt examples looked better than base, but this remains data-only full-mask bridge. | H20 dpo-diag found; local visual folder exists. |
| Exp7 | Partial-mask task alignment is necessary. Stage1 improved true partial-mask metrics over base, Stage2 DPO regressed. | Local visual folders exist; full training dpo-diag not found in current H20 scan. PAI/NAS search needed. |
| Exp9 | Target-domain D3 comp ckpt500 is current best DPO-S1 candidate; longer training gets worse. | H20 nocomp/no-lose dpo-diag found; PAI clean comp dpo-diag/eval artifacts need manual PAI search. |

## dpo-diag Interpretation

The result narrative must never rely on visual videos alone:

- Old Exp5 collapse requires dpo-diag showing high ranking accuracy / low DPO loss with winner damage.
- New Exp5/New Exp6 need dpo-diag to show whether winner anchoring reduced `mse_w_over_ref_mse_w`.
- Exp9 ckpt500 early-window claim needs both target metrics and dpo-diag step comparison.

See `PRD/13_dpo_diag_audit.md`.

