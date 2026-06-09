# Experiment Registry

Each experiment has an independent folder with config, paths, commands, dpo diagnostics, metric notes, qualitative notes, and status. The registry stores pointers only; videos, checkpoints, datasets, and large logs remain outside git.

Current active ledger: `current_active.md`.

Legacy entries that should not appear in the active experiment surface have been
moved, without deletion, to `../pending_delete/legacy_experiment_registry/`.

## Evidence Policy

Every DPO experiment needs `dpo_diagnostics.csv` or an explicit `MISSING_DPO_DIAG` / `REMOTE_DIAG_PATH_FOUND` status. Qualitative videos are not enough for final conclusions. Current conclusions must be supported by qualitative review, task-appropriate metrics, and dpo_diag together.

## Corrected Lineage

- Exp4: fullmask generated loser quality gate failed; no useful DPO training.
- Old Exp5: D2 comp + plain beta500 DPO; collapsed.
- Exp5 beta10 plain: beta lowered to 10, still collapsed; shows beta alone was not enough.
- New Exp5: D2 comp + winner_abs/winner_gap/lose_gap=0.25; improved relative to collapse, not final.
- New Exp6: D2 no-comp + same New Exp5 loss; not plain Exp6.
- Exp7: changed task to partial-mask inpainting; current results are suspicious and need small-mask + ProPainter-prior fix.
- Exp8: D3/YouTube-VOS region-loss diagnostic; only valid after implementation/baseline checks.
- Exp8a: completed PAI full-loss DAVIS baseline; negative result, not region-loss, not a success.
- Exp8c: completed/diagnostic YouTube-VOS GT-winner + D3 loser full-loss pair.
- Exp9: completed log-ratio / normalized-gap DPO; user-facing split is Exp9-1
  for Stage1 DPO + SFT Stage2 validation and Exp9-2 for Stage1 DPO + Stage2
  DPO validation.
- Exp10: region-local DPO; Exp10-1 is partially complete on PAI. Fresh retries
  must restart from scratch rather than resuming the interrupted checkpoint.
- Exp11: blocked until train-time flow/prior consistency audit passes.

See `experiment_matrix.md` for the full table.
