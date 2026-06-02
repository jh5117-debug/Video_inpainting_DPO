# DPO-S1 + SFT-S2 Hybrid Plan

Updated: 2026-06-02

## Motivation

Exp7-PM-Gate1500 changed the interpretation of the checkpoint results:

- `Stage1_last` improves true partial-mask metrics over DiffuEraser-base.
- `Stage2_last` regresses below both `Stage1_last` and DiffuEraser-base.
- Full-mask qual30 remains failed / task-mismatched for Exp7.

DiffuEraser is a two-stage model:

| Stage | Primary role |
| --- | --- |
| Stage1 | spatial generation quality, BrushNet, UNet2D, appearance |
| Stage2 | temporal consistency, motion module, temporal modeling |

Therefore the next candidate is not "Stage1-only inference." The correct
candidate is:

```text
DPO Stage1 spatial / appearance weights
+
frozen SFT Stage2 temporal / motion weights
```

## Checkpoint Rule

Do not simply load a full SFT Stage2 checkpoint over the DPO result. A Stage2
export can contain both spatial and temporal weights, so a naive overwrite can
erase the useful DPO Stage1 spatial adaptation.

The safe policy is:

```text
from DPO Stage1:
  brushnet
  UNet2D / spatial modules
  conv_in / conv_out
  down_blocks / up_blocks / mid_block non-motion modules

from SFT Stage2:
  UNetMotionModel temporal / motion modules
  motion_modules
  temporal_transformer / temporal_attention keys
```

The existing DiffuEraser Stage2 code already follows this pattern: load a
motion UNet, copy Stage1 2D modules into it, and load BrushNet from Stage1.

## Tooling

New tools:

```text
tools/inspect_diffueraser_stage_weights.py
tools/build_diffueraser_dpoS1_sftS2_hybrid.py
scripts/eval_exp7_dpoS1_sftS2_hybrid_partialmask.sh
scripts/launch_exp7_pm_stage1only_ckptsweep_pai.sh
```

`inspect_diffueraser_stage_weights.py` writes:

```text
reports/diffueraser_stage_checkpoint_structure.md
```

It records Stage1/Stage2 files, config classes, sample keys, and the spatial
vs temporal/motion split.

`build_diffueraser_dpoS1_sftS2_hybrid.py` builds:

```text
output_dir/
  last_weights/
    unet_main/
    brushnet/
  hybrid_manifest.json
  key_merge_report.json
  key_merge_report.md
```

It supports dry-run and reports:

- `loaded_from_dpo_stage1`
- `loaded_from_sft_stage2`
- `skipped`
- `shape_mismatch`
- `unexpected`
- `missing`
- `uncertain_preserved_from_sft_stage2`

## Hybrid Eval

Script:

```text
scripts/eval_exp7_dpoS1_sftS2_hybrid_partialmask.sh
```

Experiment name:

```text
exp7_pm_dpoS1_sftS2_hybrid_ckptsweep
```

Data:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4/manifests/selected_primary_comp.repaired.jsonl
```

DPO Stage1 candidates:

- `checkpoint-500`
- `checkpoint-1000`
- `checkpoint-1500`
- `checkpoint-2000`
- `checkpoint-2500`
- `checkpoint-3000`
- `last_weights`

Missing or non-exported checkpoints are skipped and recorded.

SFT Stage2 candidate priority:

1. YouTube-VOS SFT Stage2 if found.
2. User-specified / previous SFT Stage2 if found.
3. Official/base DiffuEraser Stage2 converted weights.

Comparisons:

- DiffuEraser-base.
- Exp7 DPO Stage1_last as a diagnostic direct Stage1 export if evaluable.
- Exp7 DPO Stage1 + DPO Stage2 last if evaluable.
- Every built DPO-S1 + SFT-S2 hybrid.

Output:

```text
logs/partialmask_eval/exp7_pm_dpoS1_sftS2_hybrid_<timestamp>/
  metrics/summary.csv
  metrics/summary.json
  metrics/summary.md
  side_by_side/
  index.html
  pair_manifest.csv
  hybrid_reports/
```

Final report:

```text
reports/exp7_dpoS1_sftS2_hybrid_eval_report.md
```

## Decisions To Make From The Report

The report must answer:

1. Can DPO Stage1 and SFT Stage2 be safely hybridized?
2. Which SFT Stage2 checkpoint was used?
3. Was YouTube-VOS SFT Stage2 found?
4. Were DPO Stage1 spatial weights preserved?
5. Were SFT Stage2 motion weights preserved?
6. Which DPO Stage1 checkpoint + SFT Stage2 combination is best?
7. Does the hybrid beat DiffuEraser-base?
8. Does the hybrid beat Exp7 DPO Stage1 + DPO Stage2?
9. Should DPO Stage2 remain stopped?
10. Is the next move Stage1-only checkpoint sweep, no-lose-gap Stage1, Exp8
    region loss, or YouTube-VOS D3?

## Prepared But Not Launched

Stage1-only checkpoint sweep:

```text
scripts/launch_exp7_pm_stage1only_ckptsweep_pai.sh
```

Configuration:

```text
EXP_NAME=exp7_pm_stage1only_ckptsweep_wingap_lose025_beta10
TRAIN_MASK_MODE=partial
MASK_FROM_MANIFEST=true
LOSS_REGION_MODE=full
BETA_DPO=10
WINNER_ABS_REG_WEIGHT=0.05
WINNER_GAP_REG_WEIGHT=1.0
WINNER_GAP_REG_MARGIN=0.0
LOSE_GAP_WEIGHT=0.25
SFT_REG_WEIGHT=0.0
STAGE1_MAX_STEPS=3000
CKPT_STEPS=500
CKPT_LIMIT=10
VAL_STEPS=999999
```

This script must not start DPO Stage2 or full VBench. It should only be used
if the hybrid audit shows that more DPO Stage1 checkpoints are needed.
## 2026-06-02 Boundary Update

This hybrid plan remains useful as a bridge-domain diagnostic, but it is not the
final target-domain evaluation. The final target domains are YouTube-VOS and
DAVIS.

Current sampled-video interpretation:

- `DiffuEraser-base` is visually stronger than Exp7 variants on sampled
  partial-mask examples.
- `Exp7_DPO_Stage1_last` improves some mask metrics but shows visible flicker
  and unstable high-frequency artifacts.
- `Exp7_DPO_S1_DPO_S2_last` and the official/base Stage2 hybrid do not solve
  the quality issue and can preserve wrong-object priors.

Therefore:

- keep DPO Stage2 stopped;
- do not jump directly to Exp8;
- do not do VideoDPO partial-mask SFT warmup;
- run target-domain eval on YouTube-VOS / DAVIS before deciding Exp9.

D3 YouTube-VOS generated-loser data may sync to PAI in the background, but it is
only data preparation until target-domain eval says Exp9 is needed.
