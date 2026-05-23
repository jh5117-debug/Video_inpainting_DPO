# official_videodpo_diffueraser_data_partialmask_loser_comp

Purpose: the cleanest partial-mask data-only loser ablation.

## Definition

Changed:

- Loser is generated offline with a partial mask.
- The raw output is composited onto the winner so mask-outside pixels remain the winner exactly.

Not changed:

- Win remains the original VideoDPO winner.
- Training still uses the official VideoDPO / DiffuEraser full-mask bridge.
- The partial mask is not passed to the model during training.
- DPO loss, dataset semantics, and metric semantics are not changed.

This is a **data-only ablation**, not a task ablation.

## Key Formula

```text
win = VideoDPO winner
partial_mask = used only for offline loser generation
raw_loser = video_inpainting_model(win, partial_mask)
comp_loser = win * (1 - partial_mask) + raw_loser * partial_mask
```

Training:

```text
DiffuEraser still receives full mask / black condition.
partial_mask is not a training input.
This is a data-only ablation, not a task ablation.
```

The formula is semantic. Implementation must normalize mask polarity according to the generating model so that mask-outside pixels come exactly from `win`, and mask-inside pixels come from `raw_loser`.

## Data Contract

| Field | Value |
| --- | --- |
| win | VideoDPO winner |
| raw_loser | `video_inpainting_model(win, partial_mask)` |
| final_loser | `comp_loser` |
| mask for loser generation | partial |
| mask for training | full |
| comp | true |
| loser generation | offline |
| changed variable | data only |

Save both mask metadata and raw/comp loser paths in the manifest so the task-partialmask experiment can reuse them.

## Priority

This is the first-priority partial-mask experiment: `offline + comp`.

Reasons:

- reproducible;
- training speed remains stable;
- generation cost does not enter DPO training;
- win and loser only differ inside the mask.
