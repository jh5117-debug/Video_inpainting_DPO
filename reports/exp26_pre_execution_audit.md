# Exp26 Pre-Execution Audit

## Source

- branch: `research/exp26-videopainter-dpo-v2`
- HAL worktree: `/home/hj/H20_Video_inpainting_DPO_exp26_videopainter`
- source copied from: `exp14_adapter_videopainter/`

## Exp14 Issues Carried Into v2 Scope

The Exp14 report shows that gate2000 completed but underperformed the
VideoPainter baseline. It also recorded DPO saturation, high implicit accuracy,
and loser-dominant behavior. The copied trainer additionally had implementation
issues that must be fixed before any v2 training:

- optimizer was `AdamW(policy_branch.parameters(), lr=...)` without official
  optimizer field parity;
- `noised_image_dropout` was parsed but not used;
- first-frame GT conditioning did not force loser first frame consistency;
- old D3 16-frame input was trimmed to 13 frames, while formal VideoPainter
  evaluation should use the native 49-frame path;
- `itertools.cycle(loader)` could cache batches indefinitely;
- loser-dominant diagnostics used `m_l > m_w` instead of the formal project
  definition.

## Current Action

Exp26 now adds static fixes and unit tests for those issues. No self-loser
generation, GPU training, or DAVIS50 evaluation has started.
