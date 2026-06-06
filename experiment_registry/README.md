# Experiment Registry

Each experiment has an independent folder with config, paths, commands, dpo diagnostics, metric notes, qualitative notes, and status. The registry stores pointers only; videos, checkpoints, datasets, and large logs remain outside git.

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
- Exp9: D3 target-domain gate family; ckpt500 early window is promising but long DPO degrades.

See `experiment_matrix.md` for the full table.
