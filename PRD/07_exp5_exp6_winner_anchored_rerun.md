# Exp5 / Exp6 Winner-Anchored Rerun

Updated: 2026-06-02

## Failed Diagnostic Runs

`exp5_d2_comp_k4_stage1/stage2_full` with `beta_dpo=500` is failed /
collapsed / diagnostic only.

`exp5_d2_comp_k4_beta10_s1s2_4000` is also failed / collapsed / diagnostic
only.

The failure is not a task failure, not a Stage2 weight handoff bug, and not a
VBench runner bug. Stage2 loaded Stage1 `last_weights` correctly and completed.
The failure is the unanchored D2 generated-loser + full-mask/full-video DPO
objective.

Observed pattern:

- `mse_w >> ref_mse_w`: policy gets worse on the winner.
- `mse_l >> ref_mse_l`: policy gets worse on the loser.
- `lose_gap` grows more than the winner damage, so the DPO ranking is satisfied.
- `implicit_acc` approaches 1 and `dpo_loss` approaches 0.
- Visual output collapses into universal stripe / high-frequency noise / color
  explosion.

## New Objective

Winner-anchored DPO adds explicit pressure to keep the winner prediction close
to the reference:

```text
L_total = L_DPO
        + lambda_abs * m_w
        + lambda_gap * ReLU(m_w - m_w_ref - margin)
```

Where:

- `m_w` is policy winner MSE.
- `m_w_ref` is reference winner MSE.
- `ReLU(m_w - m_w_ref - margin)` directly suppresses winner-gap explosion.

The default values remain zero, so older experiments keep the old behavior.

## New Exp5

```text
name = exp5_d2_comp_k4_wingap_lose025_beta10_s1s2_4000
manifest = selected_primary_comp.repaired.jsonl
train_mask_mode = full
mask_from_manifest = false
loss_region_mode = full
beta_dpo = 10
winner_abs_reg_weight = 0.05
winner_gap_reg_weight = 1.0
winner_gap_reg_margin = 0.0
lose_gap_weight = 0.25
sft_reg_weight = 0.0
stage1_steps = 4000
stage2_steps = 4000
post_stage2_eval = qual30 + full VBench
```

## New Exp6

```text
name = exp6_d2_nocomp_k4_wingap_lose025_beta10_s1s2_4000
manifest = selected_primary_nocomp.repaired.jsonl
train_mask_mode = full
mask_from_manifest = false
loss_region_mode = full
beta_dpo = 10
winner_abs_reg_weight = 0.05
winner_gap_reg_weight = 1.0
winner_gap_reg_margin = 0.0
lose_gap_weight = 0.25
sft_reg_weight = 0.0
stage1_steps = 4000
stage2_steps = 4000
post_stage2_eval = qual30 + full VBench
```

## H20 Exp6 Policy

All active old Exp6 unanchored training processes should be stopped and marked
superseded. Do not delete old outputs; keep them for failure audit evidence.

Current H20 winner-anchored Exp6:

```text
name = exp6_d2_nocomp_k4_wingap_lose025_beta10_s1s2_4000
status = running / continue
action = monitor only; do not kill
purpose = no-comp data-only comparison against Exp5 comp
```

2026-06-02 monitor snapshot:

```text
current_stage = Stage2
stage1_completed = 2026-06-01 22:51 CST
stage2_progress = about 420 / 4000
num_gpus = 6
gpu_policy = GPU 0-5 only; GPU 6/7 idle
stage2_loaded = Stage1 last_weights
qual30 = pending
full_vbench = pending
```

Stage2 diagnostics show the winner anchor is active (`winner_gap_reg` near
zero; `mse_w_over_ref_mse_w` near 1), but the loser shortcut remains strong
(`loser_dominant_ratio=1.0`, high `mse_l_over_ref_mse_l`, high `sigma_term`).
This is still the intended no-comp comparison run and should continue.

## Exp7 Partial-Mask Gate

Exp5 winner-anchored is improved but not final. The winner anchor suppressed
the worst unanchored collapse, but the run still shows texture/color attractors
under a data-only full-mask/full-video objective.

Exp7 tests task alignment:

```text
name = exp7_d2_comp_k4_partial_wingap_lose025_beta10_s1s2_gate1500
manifest = selected_primary_comp.repaired.jsonl
train_mask_mode = partial
mask_from_manifest = true
M_train = M_gen from manifest mask_path
loss_region_mode = full
beta_dpo = 10
winner_abs_reg_weight = 0.05
winner_gap_reg_weight = 1.0
winner_gap_reg_margin = 0.0
lose_gap_weight = 0.25
sft_reg_weight = 0.0
stage1_steps = 1500
stage2_steps = 1500
validation = qual30 side-by-side + dpo_diag summary
full_vbench = disabled by default for gate
```

Exp5/6 are still data-only full-mask bridge experiments. Exp7 is a task gate:
training uses the generated loser manifest mask as DiffuEraser's partial-mask
condition. If Exp7 is more stable, the D2 data itself is likely usable and the
main failure was objective/task mismatch.

## 2026-06-01 Exp7 Gate1500 Status

Exp7 gate1500 completed Stage1 and Stage2 with:

```text
train_mask_mode = partial
mask_from_manifest = true
beta_dpo = 10
winner_abs_reg_weight = 0.05
winner_gap_reg_weight = 1.0
winner_gap_reg_margin = 0.0
lose_gap_weight = 0.25
stage1_steps = 1500
stage2_steps = 1500
```

The current full-mask-style qual30 is poor: many examples are stripe-heavy, and
some look worse than the new Exp5 winner-anchored run. This is **not a fair
final Exp7 evaluation** because Exp7 is a partial-mask task run, while that
qualitative eval is still full-mask prompt generation.

Current interpretation:

- full-mask qual30 = failed / task-mismatched;
- Exp7 gate status = inconclusive / risky;
- partial-mask manifest evaluation is pending;
- do not launch full Exp7 4000+4000, full VBench, or Exp8 from this result.

Diagnostic readout so far:

- Winner-gap regularization is doing useful work and keeps `win_gap` bounded
  relative to the unanchored Exp5 collapse.
- Loser degradation remains strong: `loser_dominant_ratio` reaches 1.0 and
  `mse_l_over_ref_mse_l` can become very high.
- The required next check is true partial-mask inpainting eval using D2
  `win_video_path` and `mask_path`.

Prepared fallback gate, not launched:

```text
name = exp7_d2_comp_k4_partial_wingap_nolose_beta10_s1s2_gate1000
train_mask_mode = partial
mask_from_manifest = true
loss_region_mode = full
beta_dpo = 10
winner_abs_reg_weight = 0.05
winner_gap_reg_weight = 1.0
winner_gap_reg_margin = 0.0
lose_gap_weight = 0.0
stage1_steps = 1000
stage2_steps = 1000
```

## 2026-06-02 Exp7-PM-Gate1500 Partial-Mask Eval

The task-matched partial-mask evaluation completed after replacing the eval
tool's fragile `imageio` video reader with ffmpeg rawvideo decoding.

```text
eval_name = Exp7-PM-Gate1500
output_root = /mnt/nas/hj/H20_Video_inpainting_DPO/logs/partialmask_eval/exp7_gate1500_20260602_000500
side_by_side = 60 videos
metrics = metrics/summary.csv
report = report.md
```

Evaluated checkpoints:

| Checkpoint | Status |
| --- | --- |
| DiffuEraser-base | active |
| Stage1_ckpt500 | skipped; exported path missing |
| Stage1_ckpt1000 | skipped; exported path missing |
| Stage1_last | active |
| Stage2_last | active |

Metric result:

| Model | mask_region_psnr_mean | mask_region_ssim_mean | outside_region_diff_mean_mean | temporal_diff_delta_vs_gt_mean |
| --- | ---: | ---: | ---: | ---: |
| DiffuEraser-base | 8.99765 | 0.272146 | 2.91477 | 5.58378 |
| Stage1_last | 9.57079 | 0.288404 | 2.92006 | 12.7824 |
| Stage2_last | 7.88448 | 0.235938 | 2.91600 | 6.52143 |

Interpretation:

- The full-mask qual30 failure was task-mismatched and should not be the final
  Exp7 verdict.
- On the true partial-mask inpainting eval, Exp7 `Stage1_last` beats
  DiffuEraser-base by mask-region PSNR and SSIM.
- `Stage2_last` regresses below both `Stage1_last` and DiffuEraser-base.
- Exp7 validates the partial-mask task-alignment direction, but not the current
  Stage1+Stage2 recipe.
- Do not launch full Exp7 4000+4000 yet.
- Next decision should compare visual side-by-side quality and likely test
  the prepared no-lose-gap gate if Stage2 artifacts match loser-degradation.
