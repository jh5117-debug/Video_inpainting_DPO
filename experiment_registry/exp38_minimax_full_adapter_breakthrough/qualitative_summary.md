# Exp38 Qualitative Summary

No Exp38 videos have been generated yet.

Readback imported prior MiniMax visual status:

- Exp30: `0/32` visual better comparisons.
- Exp35: `0/48` visual better comparisons.
- Exp36: `0/24` visual better comparisons.
- Exp37: each LocalDPO-badnoise recipe had only `1/16` visual better heldout
  rows, below the positive gate.

Future Exp38 pass/positive statuses require generated videos, temporal strips,
per-video review, metrics, and explicit artifact/outside-damage reporting.

## 2026-06-28 Failure Taxonomy

No videos were generated. The taxonomy preserves the prior visual gate:

- Visible promotion is not allowed from metric-only movement.
- R1 from Exp37 remains a useful clue but not a pass because only one heldout
  row visibly improved.
- The next visual question is whether existing/bounded positive-controls can
  visibly improve training videos before any heldout DPO expansion.

## 2026-06-28 Train-Overfit Diagnosis

Codex reviewed compact temporal strips for Exp37 R1 train32/heldout16 and Exp36
S1 train32/heldout16. Each compact strip covers start/mid/end frames with
Step0, Step10, and diff columns; representative full 16-frame strips were also
opened.

Visual conclusion:

- Exp37 R1: visible output movement exists, but train samples often show
  outside/global drift. Heldout samples are mixed and do not show a reliable
  quality gain.
- Exp36 S1: train and heldout outputs are visually near-identical to Step0.

No `QUALITY_POSITIVE`, `THIRD_BACKBONE`, or `UNIVERSAL_ADAPTER` language is
unlocked.
