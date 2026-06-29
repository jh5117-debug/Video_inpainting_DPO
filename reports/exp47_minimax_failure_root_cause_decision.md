# Exp47 MiniMax Failure Root Cause Decision

Final status: `EXP47_FORENSIC_DECISION_COMPLETE`
Final root cause: `GLOBAL_SFT_SHOULD_BE_LOCALIZED`
Final next step: `NEXT_H20_LOCAL_PSEUDOSUCCESS_TARGET_1_10STEP`

## Evidence

1. Manifest/path/frame alignment passed: `EXP47_MANIFEST_ALIGNMENT_PASS` with `112` rows and `0` failures. Active target was Exp46 `winner_path` pseudo-success frames, not GT/failure/condition.
2. Teacher quality is not global-clean: `EXP47_TEACHER_GLOBAL_DRIFT_CONFIRMED`. Strict-clean count is `0`; teacher global-drift rows are `26` and local-only rows are `22` under Milestone C labels.
3. Step30 movement does not approach the pseudo target or V_bg: closer-to-pseudo `0/48`, closer-to-V_bg `0/48`. This rules out a simple "learned bad teacher" explanation as the only cause.
4. Region loss is too global for local-only pseudo-success: `EXP47_REGION_LOSS_GLOBAL_DRIFT_RISK_CONFIRMED`. Outside + affected + global-base component contribution is `0.824424` overall and `0.912822` on the search proxy.
5. Strict global pseudo-success relabel fails: strict-clean `0`, local-only `48`, strict 32/16/16 split possible `False`.

## Required Answers

1. Was Exp46 negative because target teacher was bad? Partly: the pseudo-success teacher is bad as a global SFT target, but still has local removal signal.
2. Was it a manifest/path/frame bug? No. Milestone B passed with zero failed rows.
3. Was it a region loss/mask bug? Mask polarity and finite weights pass; the issue is objective localization/global contribution risk, not a simple polarity bug.
4. Did Step30 move toward pseudo target? No. It moved closer to pseudo target in `0/48` sampled rows.
5. Can pseudo-success still be used globally? No.
6. Can pseudo-success be used locally? Yes, after constructing localized targets/loss and only with a tiny 1/10-step probe.
7. Should we run any more SFT? Not global pseudo-success SFT. Only localized pseudo-success 1/10-step after target construction.
8. Should we run same-source DPO next? Not first. Same-source DPO remains plausible, but the immediate next step is localized pseudo-success target construction.
9. Is MiniMax third adapter evidence? No.
10. What exactly should PAI or H20 do next? H20 should build a local pseudo-success target/loss from `manifests/exp47_success_local_only.jsonl` and run only a 1/10-step probe; PAI may later re-mine strict global-clean pseudo-success if needed.
