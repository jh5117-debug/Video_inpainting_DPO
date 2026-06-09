# Current Active Experiment Ledger

Updated: 2026-06-09

This file is the compact source of truth for the experiments that should remain
visible in the active registry/code structure. Older exploratory gates are moved
to `pending_delete/` unless they are needed to explain one of these rows.

| User-facing name | Registry / folder | Status | Notes |
| --- | --- | --- | --- |
| pre-Exp5 historical setup | `exp01_*`, `exp02_*`, `exp03_*`, `exp04_*` | historical context | Keep for lineage only. |
| Exp5 | `exp05_old_d2_comp_plain_failed`, `exp05_beta10_plain_failed` | failed diagnostics | Plain DPO collapse evidence. |
| NewExp5 | `exp05_new_d2_comp_wingap_lose025` | completed, improved but not final | Winner anchoring helped. |
| NewExp6 | `exp06_new_d2_nocomp_wingap_lose025` | completed | No-comp winner anchored diagnostic. |
| Exp7a-1 | `exp07_fix_smallmask_prior` / `exp07_current` evidence | completed/evaluated | Small-D2 / prior corrected validation family. |
| Exp7a-2 | `exp07_fix_smallmask_prior` / `exp07_current` evidence | completed/evaluated | Second Exp7a validation variant. |
| Exp8a-1 | `exp08a_d3_comp_fullloss_davis_s1s2_2000` | completed negative | Stage1 DPO + SFT Stage2 DAVIS. |
| Exp8a-2 | `exp08a_d3_comp_fullloss_davis_s1s2_2000` | completed negative | Stage1 DPO + Stage2 DPO DAVIS. |
| Exp8c-1 | `exp08c_youtubevos_gtwin_d3comp_fullloss_davis_s1s2_2000` | completed/diagnostic | GT winner full-loss Stage1 validation family. |
| Exp8c-2 | `exp08c_youtubevos_gtwin_d3comp_fullloss_davis_s1s2_2000` | completed/diagnostic | GT winner full-loss Stage2 validation family. |
| Exp9-1 | `exp09_logratio_gap_dpo` | completed | Log-ratio normalized gap, Stage1 DPO + SFT Stage2 DAVIS. |
| Exp9-2 | `exp09_logratio_gap_dpo` | completed | Log-ratio normalized gap, Stage1 DPO + Stage2 DPO DAVIS. |
| Exp10-1 | `exp10_region_local_dpo` | blocked by external SIGTERM | Fresh no-resume retry also failed at `2026-06-09 14:55:32 CST`; not a checkpoint-only issue. |
| Exp11 | `exp11_flow_prior_consistency_dpo` | blocked | Do not run until train-time flow/prior audit passes. |

Rules:

- New target-domain experiments use numbered names Exp9, Exp10, Exp11. Do not
  create Exp9a/Exp10a names.
- Exp10 fresh no-resume/no-policy-init was tested and still received external
  SIGTERM. Do not keep relaunching until the PAI sender/policy is identified.
- Exp11 must not be launched as a fake Exp10 clone. It requires a passed
  implementation audit for train-time flow/prior consistency.
- DAVIS validation must use 24 effective frames (`DAVIS_VIDEO_LENGTH=24`) even
  though existing D3 generated-loser training clips are 16-frame clips.
