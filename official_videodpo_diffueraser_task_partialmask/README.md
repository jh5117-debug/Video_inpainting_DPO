# official_videodpo_diffueraser_task_partialmask

Purpose: task ablation where DiffuEraser is trained as a partial-mask video inpainting model rather than through the full-mask bridge.

## Definition

Changed:

- Training mask is partial.
- DiffuEraser receives the partial mask as its task condition.
- Data can reuse the win / loser / mask outputs from `official_videodpo_diffueraser_data_partialmask_loser_comp_k4`.

Not changed in first scaffold:

- DPO loss math is not changed.
- Existing metrics are not changed.
- Same-mask is the first supported design target; resampled/mixed masks are TODO.

## Mask Policy

First version:

```text
same-mask: training mask == mask used to generate loser
```

Reserved future policies:

```text
resampled-mask: training mask is sampled independently
mixed-mask: part same-mask, part resampled-mask
```

## Data Contract

| Field | Value |
| --- | --- |
| win | VideoDPO winner |
| loser | preferably `comp_loser` from partialmask_comp data |
| mask for loser generation | partial |
| mask for training | partial |
| comp | inherited from selected data, first target is comp |
| changed variable | task |

## Important Difference From Data-Only Ablations

In data-only partial-mask experiments, partial mask is only used offline and training still uses full mask. In this task ablation, partial mask is part of the model input during training.

Therefore this is the first experiment in the plan that changes the **task**, not just the data.

Loss-region variants such as mask-only, boundary-weighted, and outside-region penalties are documented in `PRD/04_metrics_and_diagnostics.md`; they should not be enabled by default in this scaffold.
