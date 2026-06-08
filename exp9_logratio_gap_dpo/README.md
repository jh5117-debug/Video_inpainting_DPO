# Exp9 Log-Ratio Gap DPO

Experiment name: `exp9_logratio_gap_dpo_s1s2_2000_davis_pai`

Purpose: fix raw win-gap / lose-gap scale mismatch while preserving pairwise DPO. Exp9 uses log-ratio normalized gaps and clips the loser gap at `LOSE_GAP_CLIP_TAU=1.0`; it does not remove the loser branch.

Pipeline:

1. Stage1 DPO, 2000 steps.
2. DAVIS validation for `DPO-S1 + SFT-S2`.
3. Stage2 DPO, 2000 steps.
4. DAVIS validation for `DPO-S1 + DPO-S2`.
5. DPO diagnostics and final report.

Default PAI master launch runs this experiment only.
