# Exp25 CLI4 DE-B Gate16 Result

Status: `EXP25_DIFFUSERASER_GATE16_PASSED`

Run:

```text
gate16_deb_20260625_184632
```

Fixed stack:

```text
DE-B_sft_raw6_d8_propainter
pcm_mode = none
prior_mode = propainter
no_pcm_steps = 6
guidance = 0.0
mask_dilation_iter = 8
hard_comp = false
condition = V_obj
winner = V_bg
mask = object foreground mask
loser = DiffuEraser raw output
```

Selection:

- Source pool: VOR train source pool.
- Excluded root-cause 12, VOR-Eval, search-dev, shadow-dev, and previous
  Gate32 scene groups.
- Scene groups are disjoint inside Gate16.
- Best available balance after exclusions was 12 REAL and 4 BLENDER. Only four
  BLENDER rows remained after all disjointness filters.

Result summary:

| item | value |
| --- | ---: |
| generated raw losers | 16 / 16 |
| technical valid | 16 / 16 |
| medium-hard eligible | 7 / 16 |
| hard but plausible | 7 / 16 |
| too close | 0 / 16 |
| trivial bad | 2 / 16 |
| technical invalid | 0 / 16 |
| mean mask PSNR | 17.384136 |
| mean outside PSNR | 22.776255 |
| mean temporal absdiff | 5.261560 |

Gate decision:

```text
EXP25_DIFFUSERASER_GATE16_PASSED
```

Reason: technical valid is 16/16, medium-hard plus hard-plausible is 14/16,
trivial-bad is 2/16, and dense evidence review reports no system outside
collapse.

Per-sample classifications:

| sample | source | class |
| --- | --- | --- |
| REAL_ENV157_00014_005_01 | REAL | MEDIUM_HARD_ELIGIBLE |
| REAL_ENV180_00011_007_03 | REAL | HARD_BUT_PLAUSIBLE |
| REAL_ENV203_00008_002_05 | REAL | MEDIUM_HARD_ELIGIBLE |
| REAL_ENV204_00016_002_05 | REAL | TRIVIAL_BAD |
| REAL_ENV175_00015_004_05 | REAL | MEDIUM_HARD_ELIGIBLE |
| REAL_ENV164_00004_002_04 | REAL | HARD_BUT_PLAUSIBLE |
| REAL_ENV282_00102_005_04 | REAL | MEDIUM_HARD_ELIGIBLE |
| REAL_ENV232_00101_006_05 | REAL | TRIVIAL_BAD |
| BLENDER_STREET001_00003 | BLENDER | MEDIUM_HARD_ELIGIBLE |
| BLENDER_BEDROOM007_00002 | BLENDER | HARD_BUT_PLAUSIBLE |
| BLENDER_BEDROOM004_00005 | BLENDER | MEDIUM_HARD_ELIGIBLE |
| BLENDER_WAREHOUSE034_00002 | BLENDER | MEDIUM_HARD_ELIGIBLE |
| REAL_ENV210_00016_005_05 | REAL | HARD_BUT_PLAUSIBLE |
| REAL_ENV266_00105_001_01 | REAL | HARD_BUT_PLAUSIBLE |
| REAL_ENV270_00102_001_03 | REAL | HARD_BUT_PLAUSIBLE |
| REAL_ENV236_00101_002_03 | REAL | HARD_BUT_PLAUSIBLE |

Evidence roots:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_vor_or_preference_data/cli4/gate16_deb_20260625_184632
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp25_vor_or_preference_data/cli4/gate16_deb_20260625_184632
```

Evidence files:

- `review_v2/review_summary.json`
- `reports/exp25_diffueraser_or_root_cause_matrix_v2.md`
- `reports/exp25_diffueraser_or_root_cause_visual_review.csv`
- `review_v2/dense_contact_sheets/DE-B_sft_raw6_d8_propainter/*.jpg`
- `review_v2/mask_crops/DE-B_sft_raw6_d8_propainter/*.jpg`
- `review_v2/animated_gif/DE-B_sft_raw6_d8_propainter/*.gif`
- `review_v2/affected_crops/DE-B_sft_raw6_d8_propainter/*.jpg`
- `review_v2/outside_crops/DE-B_sft_raw6_d8_propainter/*.jpg`
- `review_v2/temporal_difference_top3/DE-B_sft_raw6_d8_propainter/*.jpg`

CLI4 manual visual review:

- Opened all 16 dense temporal contact sheets locally after copying them from
  the PAI run root.
- Generated supplemental affected/outside/top3 crop sheets from the existing
  winner/raw/mask PNG frames after aligning winner/mask to the raw prediction
  resolution.
- The manual review agrees with the automatic classes: 7 medium-hard, 7
  hard-plausible, and 2 trivial-bad.
- No sheet showed system-wide outside collapse, black-frame failure, or
  global temporal breakage. The observed failures are local residuals,
  local blur, or overly obvious retained objects.

Next action:

Build a candidate-pool and Gate64 plan. Do not start OR-DPO from this CLI
branch, and do not expand Gate128 in this round.
