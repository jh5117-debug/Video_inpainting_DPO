# Exp11 Outer B0.75 S2 Visual Evidence Verification

Date: 2026-06-15

## Verification Result

Status: **verified complete**.

No rerun is needed.

HAL archive:

```text
/home/hj/dpo-2-1-exp/this_week_exp11_exp12/visual_evidence/exp11_outer_b075_s2_selected_visuals
```

PAI source:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/20260615_exp11_outer_b075_s2_selected_visuals
```

## File Completeness

| item | status | count / path |
|---|---|---|
| side-by-side MP4 | OK | 8 files |
| selected frame JPG | OK | 40 files |
| frame contact sheets | OK | 8 files |
| manifest | OK | `visual_evidence/visual_evidence_manifest.csv` |
| README | OK | `README.md` |
| SFT-48000 per-video metrics | OK | `framewise_metric/DiffuEraser-base/metrics/per_video_metrics.csv` |
| Exp11 per-video metrics | OK | `framewise_metric/Exp11_boundary_outer_b075_S2/metrics/per_video_metrics.csv` |

## Case Review

Strongest positive:

- `boat`

Usable positives:

- `rhino`
- `dog-agility`
- `lucia`
- `blackswan`

Caution / failure cases:

- `dance-jump`
- `soccerball`

`boat` is the clearest evidence case because SFT-48000 shows a visible white fog / patch around the wake and hull while Exp11 outer b0.75 S2 keeps cleaner water texture and boundary continuity.

## Metric Check On Selected Evidence Set

Positive delta means Exp11 outer b0.75 S2 is better than SFT-48000 in the selected rerun.

| video | delta whole PSNR | delta mask PSNR | delta whole SSIM | delta mask SSIM | interpretation |
|---|---:|---:|---:|---:|---|
| boat | +1.4362 | +1.4362 | +0.0208 | +0.0925 | strong positive |
| rhino | +0.9862 | +0.9862 | +0.0052 | +0.0133 | positive |
| dog-agility | +0.7001 | +0.7001 | +0.0023 | -0.0167 | useful visual positive, mask SSIM mixed |
| blackswan | +0.4102 | +0.4102 | +0.0017 | -0.0048 | mild positive, mask SSIM mixed |
| lucia | +0.3976 | +0.3976 | +0.0008 | +0.0167 | positive but subtle |
| flamingo | +0.1811 | +0.1811 | +0.0003 | -0.0000 | weak positive |
| dance-jump | -1.0214 | -1.0214 | -0.0039 | -0.0253 | caution / failure |
| soccerball | -0.8039 | -0.8039 | -0.0090 | -0.0772 | caution / failure |

## Decision

The selected visuals are complete and usable. Do not rerun this selected visual task unless one of the archived files is lost.

