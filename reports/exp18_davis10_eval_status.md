# Exp18 DAVIS10 Eval Status

The original overnight launcher only wrote an eval guard status. A true
DAVIS10 hybrid eval was then run with:

```text
exp18_multiframe_propagation_gated_dpo/scripts/run_exp18_davis10_hybrid_eval_pai.sh
```

Output root:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp18_multiframe_propagation_gated_dpo_davis10
```

Completed methods:

| Method | Status |
|---|---|
| SFT-48000 baseline | complete |
| Exp11 boundary outer b0.75 S2 | complete |
| Exp18a prop-only S1-500 | complete |
| Exp18b prop+gen S1-500 | complete |
| Exp18c oracle S1-500 | complete |

Summary:

```text
reports/exp18_davis10_metric_summary.md
reports/exp18_visual_case_judgement.md
reports/exp18_final_pai_gate_report.md
```
