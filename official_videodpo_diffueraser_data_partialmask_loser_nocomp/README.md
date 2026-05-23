# official_videodpo_diffueraser_data_partialmask_loser_nocomp

Purpose: diagnostic comp-vs-no-comp ablation.

## Definition

Changed:

- Loser is generated offline with a partial mask.
- The raw inpainting output is used directly as the final loser.

Not changed:

- Win remains the original VideoDPO winner.
- Training still uses the official VideoDPO / DiffuEraser full-mask bridge.
- The partial mask is not passed to the model during training.
- DPO loss, dataset semantics, and metric semantics are not changed.

This is a **data-only ablation**, not a task ablation.

## Data Contract

| Field | Value |
| --- | --- |
| win | VideoDPO winner |
| raw_loser | `video_inpainting_model(win, partial_mask)` |
| final_loser | `raw_loser` |
| mask for loser generation | partial |
| mask for training | full |
| comp | false |
| loser generation | offline |
| changed variable | data only, diagnostic |

Risk: no-comp may introduce color, texture, brightness, and temporal differences outside the intended mask region. This is intentionally less controlled than the comp experiment and should be interpreted as a diagnostic ablation.

## Priority

This is second priority after `offline + comp`. It is useful for diagnosing whether compositing is necessary, but it should not replace the comp main experiment because mask-outside drift can become a preference shortcut.
