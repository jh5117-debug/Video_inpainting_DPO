# Experiment Registry And DPO-Diagnostics Audit

Updated: 2026-06-04

## Source Inputs

- PAI audit archive: `/home/hj/pai_experiment_registry_reports.tar.gz`
- PAI audit CSV: `reports/pai_experiment_registry_paths.csv` with 34,448 rows
- H20 audit CSV: `reports/h20_experiment_registry_paths.csv`
- Registry matrix: `experiment_registry/experiment_matrix.csv`

## Corrected Experiment Lineage

| Unit | Meaning | Status |
| --- | --- | --- |
| Exp4 | fullmask generated loser quality gate | failed / deleted artifact / no useful DPO |
| Old Exp5 | D2 comp + plain beta500 DPO | collapsed |
| Exp5 beta10 plain | D2 comp + beta10 but still no winner guardrail | collapsed / diagnostic |
| New Exp5 | D2 comp + beta10 + winner_abs=0.05 + winner_gap=1.0 + lose_gap=0.25 | improved but not final |
| New Exp6 | D2 no-comp + same New Exp5 loss | not plain Exp6 |
| Exp7 | partial-mask task with old masks and likely no ProPainter-prior eval | suspicious / needs fix |
| Hybrid | DPO Stage1 + frozen base/SFT Stage2 eval | did not rescue Exp7 |
| Exp8 | D3 comp region-loss diagnostic | remote diag found, implementation/eval still pending |
| Exp9 comp | D3 comp Stage1 gate | ckpt500 early-window candidate, longer DPO degrades |
| Exp9 nocomp | D3 no-comp Stage1 gate | H20 complete; caution from qualitative review |
| Exp9 no-lose | D3 comp lose_gap_weight=0 gate | local diag found; eval pending |

## Real Loss Formulas

Notation:

```text
m_w     = policy winner MSE
m_l     = policy loser MSE
m_w_ref = reference winner MSE
m_l_ref = reference loser MSE
win_gap  = m_w - m_w_ref
lose_gap = m_l - m_l_ref
inside = -0.5 * beta_dpo * (win_gap - lose_gap_weight * lose_gap)
L_DPO = mean[-logsigmoid(inside)]
L_total = L_DPO + sft_reg_weight*m_w + winner_abs_reg_weight*m_w + winner_gap_reg_weight*ReLU(win_gap - margin)
```

Key settings:

- Old Exp5: `L = -logσ{-0.5*500*(win_gap - lose_gap)}`.
- Exp5 beta10 plain: `L = -logσ{-0.5*10*(win_gap - lose_gap)}`.
- New Exp5 / New Exp6 / Exp7 / Exp9 lose025: `L = -logσ{-0.5*10*(win_gap - 0.25*lose_gap)} + 0.05*m_w + ReLU(win_gap)`.
- No-lose candidate: `L = -logσ{-0.5*10*win_gap} + 0.05*m_w + ReLU(win_gap)`.
- Exp8 region loss: planned/diagnostic until weighted-region implementation is verified from code.

## DPO-Diag Status

See `reports/all_experiments_dpo_diag_summary.md`.

- Remote PAI diagnostics found: Old Exp5, Exp5 beta10 plain, New Exp5, Exp7, Exp8, Exp9-comp.
- Local diagnostics found: New Exp6, Exp9-nocomp, Exp9 no-lose.
- Missing diagnostics: historical Exp2/Exp3 and future Exp7-fix.
- Exp4 has no useful DPO diag because it failed before a valid training run.

## Required Policy Going Forward

- Every new DPO run must start from an `experiment_registry/<exp>/` folder and config.
- Do not make final claims from qualitative videos alone.
- For partial-mask inpainting, use `inference/metrics.py` wrappers and ProPainter prior; do not use VBench.
- For YouTube-VOS/D3, use SFT-48000 DiffuEraser weights.
- Repair Exp7 with small mask 15%-20% + ProPainter prior before D3 expansion.
