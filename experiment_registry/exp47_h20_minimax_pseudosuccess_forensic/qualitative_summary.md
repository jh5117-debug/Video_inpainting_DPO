# Exp47 Qualitative Summary

Status: EXP47_READBACK_READY

Readback confirms Exp46 visual review: search has subtle global tone/haze/outside drift, while shadow has visible global brightness/color drift plus mask and boundary degradation. Exp47 will audit whether this comes from teacher quality, alignment, loss implementation, movement direction, or objective mismatch.

## Milestone B Qualitative Interpretation

The Exp46 negative result is not explained by obvious data-routing mistakes. The active target is pseudo-success, not GT, failure, or condition; frame/mask alignment passed. The forensic focus now moves to whether pseudo-success targets carry global drift and whether the loss/global target encourages that drift.

## Milestone C Qualitative Interpretation

Codex inspected the four generated teacher contact pages covering all 48 search/shadow rows. Pseudo-success often removes the local target, but many rows shift global tone/style/brightness versus `V_bg`. This supports the bad/global teacher and localization-risk hypotheses rather than a path bug.

## Milestone D Qualitative Interpretation

Movement is worse than simply learning a drifting teacher: Step30 does not get closer to either pseudo target or `V_bg` in sampled metrics. With manifest alignment already passed, this points toward region loss contribution, target velocity construction, or a MiniMax flow-objective mismatch for this pseudo-success SFT setup.

## Milestone E Qualitative Interpretation

The region loss is finite and mask polarity is sane, but it is too global for pseudo-success targets that are only local-clean. The outside region dominates normalized weight mass, `far_outside` acts as a global base, and normalized affected maps can carry pseudo target drift into outside pixels. This explains why global SFT can damage full/outside appearance even without a manifest bug.

## Milestone F Qualitative Interpretation

Strict global pseudo-success is not available. The audited pseudo-success rows should be treated as local-only signal, not global SFT targets. This pushes next-step planning toward localized pseudo-success targets or same-source preference, not another global pseudo-success SFT run.

## Milestone G Qualitative Interpretation

Exp47 finds that pseudo-success is local-only rather than global-clean. The runner/objective permits global/outside contribution, and Step30 does not move toward pseudo target or V_bg. The next repair is localized pseudo-success target construction, not more global SFT. MiniMax remains not third-backbone evidence.
