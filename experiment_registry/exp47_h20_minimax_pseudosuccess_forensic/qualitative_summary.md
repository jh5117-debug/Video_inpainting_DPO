# Exp47 Qualitative Summary

Status: EXP47_READBACK_READY

Readback confirms Exp46 visual review: search has subtle global tone/haze/outside drift, while shadow has visible global brightness/color drift plus mask and boundary degradation. Exp47 will audit whether this comes from teacher quality, alignment, loss implementation, movement direction, or objective mismatch.

## Milestone B Qualitative Interpretation

The Exp46 negative result is not explained by obvious data-routing mistakes. The active target is pseudo-success, not GT, failure, or condition; frame/mask alignment passed. The forensic focus now moves to whether pseudo-success targets carry global drift and whether the loss/global target encourages that drift.

## Milestone C Qualitative Interpretation

Codex inspected the four generated teacher contact pages covering all 48 search/shadow rows. Pseudo-success often removes the local target, but many rows shift global tone/style/brightness versus `V_bg`. This supports the bad/global teacher and localization-risk hypotheses rather than a path bug.
