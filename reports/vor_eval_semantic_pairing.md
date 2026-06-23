# VOR-Eval Semantic Pairing Audit

PAI extraction root:

`/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/extracted/vor_eval_full/VOR-Eval/VOR-Eval`

VOR-Eval was fully extracted from `VOR-Eval.tar.gz.part_000` only. No
VOR-Train archive was extracted in this step.

Observed directories:

| role | directory | count |
| --- | --- | ---: |
| winner / clean background | `BG` | 43 |
| condition / object-present video | `FG_BG` | 43 |
| foreground object mask | `MASK` | 43 |

Basename alignment:

- `BG` vs `FG_BG`: 43 overlaps, 0 missing, 0 extra.
- `BG` vs `MASK`: 43 overlaps, 0 missing, 0 extra.

Canonical OR triplet mapping:

```text
condition_video_path = FG_BG / V_obj
winner_video_path    = BG / V_bg
mask_path            = MASK / foreground object mask
task                 = object_removal
hard_comp            = false
comp_mode            = none
```

Decision:

`VOR-Eval` is a final held-out evaluation split. It must not be used for loser
generation, data-size selection, checkpoint selection, threshold tuning, loss
selection, or seed selection.
