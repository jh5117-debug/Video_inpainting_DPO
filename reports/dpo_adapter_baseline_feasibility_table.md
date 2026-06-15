# DPO Adapter Baseline Feasibility

Status: feasibility only. No adapter training launched.

Current adapter scope is narrowed to **VideoPainter** and **MiniMax-Remover** only. All other baselines are out of scope for the next adapter phase.

| Baseline | Category | Training code | Diffusion/DiT | Adapter feasibility | Action |
|---|---|---:|---:|---|---|
| VideoPainter | B needs modification | yes | yes | feasible future gate, not plug-and-play | prepare only after explicit approval |
| MiniMax-Remover | C frozen baseline | not verified | yes | no trainable adapter path validated | frozen baseline only |

Reports:

```text
reports/adapter_videopainter_feasibility.md
reports/adapter_minimax_remover_feasibility.md
reports/dpo_adapter_baseline_feasibility_table.csv
```

Do not launch adapter training without explicit user confirmation.
