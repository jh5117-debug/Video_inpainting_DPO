# PRD 50: Exp28 Fine Inner Boundary Sweep

Status: `CLI4_WAVE_RUNNING_PAIRB_REDUCED_METRIC_MIXED_NO_POSITIVE`

Branch: `research/exp28-fine-inner-boundary-sweep-20260625`

HAL worktree: `/home/hj/H20_Video_inpainting_DPO_exp28_inner_boundary`

PAI runtime snapshot target: `/home/hj/runtime_code_snapshots/cli4_exp28_<commit>`

## 2026-06-26 CLI4 Wave Evidence

Pair A (`inner2_candidate`) completed training but reduced DAVIS50 eval reached
`FAILED_FINAL` after the allowed one fix/resume cycle. Failure reason was missing
optional VFID/TC model assets before guards were added. Pair A is not evaluated
and cannot support a radius decision.

Pair B (`inner4_candidate`) completed training and reduced DAVIS50 evaluation.
The runtime eval script used optional-metric guards:

- VFID skipped: missing I3D asset.
- TC skipped: missing OpenCLIP path.
- Ewarp computed with local RAFT.

Main Stage2-2000 result:

| Metric | Delta (`candidate_s2_2000 - fresh_s2_2000`) |
| --- | ---: |
| PSNR | +0.103389 |
| SSIM | +0.000833 |
| strict mask PSNR | +0.103389 |
| mask-region SSIM | +0.007323 |
| boundary PSNR | +0.000503 |
| LPIPS | +0.000181 |
| Ewarp | -0.033830 |
| per-video PSNR win rate | 0.62 |
| bootstrap P(delta>0) | 0.9091 |

Counter-evidence:

- Stage1-hybrid 2000 is negative/mixed: PSNR -0.044981 and LPIPS +0.000462.
- VFID/TC are unavailable, so the full promotion gate is incomplete.
- Visual assets are generated for 50/50 videos, but human review is sampled only.
- Sampled visual review found no systemic outside collapse, but `surf` shows temporal-risk evidence and `cows` localized perceptual-risk evidence.

Status:

```text
INNER4_REDUCED_METRIC_MIXED
VISUAL_ASSETS_GENERATED_PARTIAL_HUMAN_REVIEW_MIXED
NO_INNER_RADIUS_POSITIVE
NO_SCIENTIFIC_POSITIVE
```

Pair C (`inner8_candidate`) is running. `fresh_control_C` Stage1 completed
checkpoint-2000 on 2026-06-26 15:14 CST and Stage2 is running.

Evidence files:

- `reports/exp28_pairB_inner4_cli4_final_decision.md`
- `reports/exp28_pairB_inner4_cli4_paired_statistics.csv`
- `reports/exp28_pairB_inner4_cli4_summary_deltas.csv`
- `reports/exp28_pairB_inner4_cli4_visual_review_human_audit.md`

## Question

On top of the already best Exp11 legacy outer one-ring boundary, does adding a narrower image-space inner boundary improve local object-removal DPO?

This experiment does not re-search the outer boundary and does not change the DPO objective or loss components beyond carving part of the mask interior into a separate inner-boundary region.

## Geometry

Mask convention: `1 = hole/object foreground`.

The outer region is fixed to the Exp11 legacy outer one latent-cell ring:

```text
outer_ring = max_pool2d(mask_loss_grid, kernel=3, stride=1, padding=1) - mask_loss_grid
```

This is recorded as `legacy outer one pool-step`. The code path for this ring is `legacy_outer_one_ring()` and it is tested to remain identical for radii 2, 4, and 8 px.

The inner region is image-space only:

```text
inner_ring_px(r) = mask_image - erode(mask_image, radius=r pixels)
```

Then it is area-pooled to the latent/loss grid. It is not built by nearest-resize thresholding.

Regions:

- `mask_core = loss_mask - inner_ring`
- `inner_ring = area_pool(mask_image - erode(mask_image, r)) clipped inside loss_mask`
- `outer_ring = legacy outer one pool-step`
- `far_outside = 1 - mask_core - inner_ring - outer_ring`

The partition must be non-negative, interpretable, and sum to 1 per loss cell within tolerance.

## Why Mixed Grids Are Allowed

The outer boundary is the known best legacy mechanism and remains on the latent/loss grid for exact Exp11 parity. The inner boundary is the new variable and is defined in image pixels because it asks whether a fine mask-interior band, not another latent max-pool ring, helps. The mixed grid is therefore deliberate and isolated: outer unchanged, inner image-space only, area-pooled once.

## Candidate Set

Main pairs:

| Pair | Fresh control | Candidate | GPUs | World size |
| --- | --- | --- | --- | --- |
| A | `fresh_control_A` | `inner2_candidate` | GPU3-4 | 2 |
| B | `fresh_control_B` | `inner4_candidate` | GPU1-2 | 2 |
| C | `fresh_control_C` | `inner8_candidate` | GPU3-4 | 2 |

Optional pair:

`inner1_candidate` is allowed only if `inner2_candidate` is positive and best.

## Weights

Main geometry phase:

```text
mask_core = 1.0
inner_boundary = 0.75
outer_boundary = 0.75
far_outside = 0.05
```

The only diagnostic exception is `INNER_WEIGHT_DIAGNOSTIC`: `r=2px`, `inner_weight=1.25`, allowed only after all 2/4/8 px radii are negative and diagnostics point to underweighting.

## Training

Every candidate has a fresh control. Historical Exp11 metrics are not accepted as the control.

For each model:

```text
Stage1 = 2000 optimizer steps
Stage2 = 2000 optimizer steps
SFT init = /mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000
Frozen reference = same SFT-48000 identity
Manifest = selected_primary_comp.gtwin.pai_paths.jsonl
Seed = 20260625
```

Exp23 used effective global batch `1 per device * grad_accum 1 * world_size 4 = 4`. Exp28 uses `world_size=2`, so the runner sets `gradient_accumulation_steps=2` to keep effective global batch exactly 4.

Stage1 and Stage2 both receive the same explicit geometry config for the main sweep. Stage2 is not allowed to silently revert to `both` or default geometry.

## Evaluation

Each pair runs fresh paired DAVIS50 after training. DAVIS50 is explicitly a model-selection benchmark here, not an untouched final test.

Protocol:

```text
432x240
24 frames
same prompts
same seed
same raw/comp protocol
same evaluator
```

Metrics:

- PSNR, SSIM, LPIPS
- VFID/FVD-style
- TC
- Ewarp
- strict mask PSNR/SSIM/LPIPS
- mask core PSNR/SSIM/LPIPS
- inner 1/2/4/8 px regional metrics
- outer legacy ring PSNR/SSIM/LPIPS
- outside preservation

Required statistics:

- candidate minus fresh control
- per-video win rate
- paired bootstrap 95% CI
- probability `delta > 0`
- leave-one-video-out range

Required visual evidence:

- side-by-side MP4 for all 50 videos
- 16-frame temporal review
- mask/core/inner/outer crops
- temporal heatmap

## Promotion Gate

An inner radius can be marked positive only if it passes all of:

- paired PSNR > +0.02 dB, or inner/boundary clearly improves while whole-video PSNR drops by no more than 0.02 dB
- per-video primary win rate >= 55%
- bootstrap probability `delta > 0` >= 0.90
- LPIPS worsening <= 0.0003
- TC drop <= 0.0002
- Ewarp worsening <= 0.03
- outer boundary not clearly worse
- no new systematic artifacts
- not dominated by one video
- checkpoint/config identity passed

Allowed statuses:

- `INNER_RADIUS_POSITIVE`
- `INNER_RADIUS_PARETO_MIXED`
- `INNER_RADIUS_NEUTRAL`
- `INNER_RADIUS_NEGATIVE`

No `INNER_RADIUS_POSITIVE` or `SCIENTIFIC_POSITIVE` claim is allowed before fresh paired DAVIS50 and visual review.

## Code and Tests

Implemented:

- `exp28_fine_inner_boundary_sweep/code/inner_boundary_geometry.py`
- `exp28_fine_inner_boundary_sweep/code/exp28_trial_runner.py`
- copied isolated Exp28 Stage1/Stage2 trainer entrypoints
- `exp28_fine_inner_boundary_sweep/scripts/eval_exp28_pair_davis50_pai.sh`

Required tests added:

- `test_inner_radius_zero_exact_control.py`
- `test_outer_ring_legacy_exact_unchanged.py`
- `test_inner_region_partition_sum_one.py`
- `test_inner_radius_pixel_geometry.py`
- `test_inner_outer_no_illegal_overlap.py`
- `test_stage1_stage2_receive_same_explicit_geometry.py`

Current validation:

```text
python -m unittest discover -s exp28_fine_inner_boundary_sweep/tests -p 'test_*.py'
python -m py_compile exp28_fine_inner_boundary_sweep/code/inner_boundary_geometry.py exp28_fine_inner_boundary_sweep/code/exp28_trial_runner.py exp28_fine_inner_boundary_sweep/code/train_exp28_stage1.py exp28_fine_inner_boundary_sweep/code/train_exp28_stage2.py exp28_fine_inner_boundary_sweep/code/summarize_exp28_pair_eval.py
bash -n exp28_fine_inner_boundary_sweep/scripts/eval_exp28_pair_davis50_pai.sh
```

All passed locally before PAI launch.
