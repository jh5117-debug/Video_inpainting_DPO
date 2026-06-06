# Status

- status: `completed_negative_davis_metrics_and_diag_present`
- conclusion: Exp8a is a completed full-loss target-domain baseline. It is not a region-loss result and it is not a success.
- evidence: Stage1 and Stage2 training completed, both DAVIS validation summaries exist, and both dpo diagnostic CSVs exist from the user-pasted PAI audit.
- result: Both `DPO-S1_SFT-S2` and `DPO-S1_DPO-S2` are substantially worse than DiffuEraser-base on DAVIS. DPO diagnostics show loser-degradation shortcut.
- next_action: Do not rerun Exp8a. Use it as negative baseline evidence while monitoring Exp8c and keeping Exp8b region-loss separate.
