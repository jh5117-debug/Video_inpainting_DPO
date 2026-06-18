# Exp18 Multi-frame Propagation-Confidence Gated DPO

Exp18 continues from the current best `Exp11 outer b0.75 S2`, but replaces the
old Exp16 GT-error prior confidence with real multi-frame propagation
confidence.

Core idea:

```text
propagatable mask pixels -> preserve propagated pixels
non-propagatable mask pixels -> generate with GT/context preference
outer boundary -> keep Exp11 boundary-aware seam constraint
```

Current status:

```text
PAI_GATE_COMPLETED_NEGATIVE_ABLATION
```

Run order on PAI:

```bash
bash exp18_multiframe_propagation_gated_dpo/scripts/launch_exp18_overnight_pai.sh
```

Result:

```text
Exp18a is the best Exp18 variant but does not beat Exp11 outer b0.75 S2 on DAVIS10.
Exp18b and Exp18c are negative.
Do not expand the current Exp18 formulation.
```

Main reports:

```text
reports/exp18_final_pai_gate_report.md
reports/exp18_davis10_metric_summary.md
reports/exp18_dpo_diag_summary.md
reports/exp18_visual_case_judgement.md
```
