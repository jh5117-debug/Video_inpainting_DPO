# Current Status

Updated: 2026-06-02

## D2 Data Readiness

The D2 generated-loser dataset is ready and must not be regenerated for the
beta10 reruns.

Root:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4
```

Ready manifests:

| Manifest | Rows |
| --- | ---: |
| `selected_primary_comp.repaired.jsonl` | 10000 |
| `selected_primary_nocomp.repaired.jsonl` | 10000 |
| `selected_secondary_comp.repaired.jsonl` | 10000 |
| `selected_secondary_nocomp.repaired.jsonl` | 10000 |

## Exp5 Collapse Status

`exp5_d2_comp_k4_stage1/stage2_full` with `beta_dpo=500` and 10000-step
Stage1/Stage2 is marked **failed / collapsed / diagnostic only**.

Evidence:

- Stage2 10000 full-mask VBench qualitative outputs show visual collapse.
- Side-by-side videos show the exp5 side as high-frequency noise and color
  explosion rather than coherent inpainting.
- DPO diagnostics saturated early with `acc=1`, `dpo=0`, and `loss=0`;
  this indicates preference-objective saturation, not image quality.
- VBench showed weak downstream behavior, including `dynamic_degree=0`,
  low `overall_consistency`, and poor scene/spatial/object dimensions.

Interpretation:

- This is not a task or code-path failure.
- Exp3 showed the VideoDPO-to-DiffuEraser DPO bridge can work.
- Exp5 failed because D2 generated losers plus full-mask training, full-video
  loss, `beta_dpo=500`, no SFT regularization, and long 10000-step training
  over-optimized the preference signal and pushed the model off distribution.
- Old Exp5 beta500 must not be used as a final result. Keep it only as a
  failed ablation and diagnostic artifact.

`exp5_d2_comp_k4_beta10_s1s2_4000` is also marked **failed / collapsed /
diagnostic only**. Its Stage2 correctly loaded Stage1 `last_weights` and
completed 4000 steps, but the visual outputs collapsed into prompt-insensitive
stripe/high-frequency textures. The log shows the same degenerate mechanism:

- `mse_w >> ref_mse_w`: the policy got worse on the winner.
- `mse_l >> ref_mse_l`: the policy got even worse on the loser.
- `win_gap` and `lose_gap` both grew, while `implicit_acc` stayed near 1.
- `dpo_loss` often approached 0, so the ranking objective was satisfied even
  though visual quality collapsed.

This confirms the failure is not a Stage2 handoff bug and not a VBench script
bug. It is a degenerate solution of D2 generated-loser + full-mask/full-video
DPO: ranking can be improved by making the loser worse while also damaging the
winner.

Replacement:

| Experiment | Status | beta_dpo | Stage1 | Stage2 | Eval |
| --- | --- | ---: | ---: | ---: | --- |
| `exp5_d2_comp_k4_wingap_lose025_beta10_s1s2_4000` | planned/running | 10 | 4000 | 4000 | qual30 + full VBench |

## H20 Old Exp6 beta500 Status

All old H20 Exp6 training using the unanchored objective is superseded and
should be stopped if still running.

Reason:

- It has the same collapse risk exposed by Exp5 beta500 and Exp5 beta10.
- It should not be continued as a final result.

Replacement:

| Experiment | Status | beta_dpo | Stage1 | Stage2 | Eval |
| --- | --- | ---: | ---: | ---: | --- |
| `exp6_d2_nocomp_k4_wingap_lose025_beta10_s1s2_4000` | planned/running | 10 | 4000 | 4000 | qual30 + full VBench |

## Current Run Policy

- Do not continue old beta500 Exp5/Exp6 or unanchored beta10 Exp5/Exp6 long
  training.
- Do not regenerate D2.
- Do not restore D1 full-mask work.
- Do not touch Exp8 region-loss settings in this pass.
- New reruns use winner-anchored DPO with `beta_dpo=10`,
  `lose_gap_weight=0.25`, `winner_abs_reg_weight=0.05`,
  `winner_gap_reg_weight=1.0`, 4000 Stage1 steps, 4000 Stage2 steps, no
  validation during training, then automatic qual30 and full VBench.

## 2026-05-31 Exp7 Partial-Mask Gate

`exp5_d2_comp_k4_wingap_lose025_beta10_s1s2_4000` is improved relative to the
unanchored Exp5 runs: winner-gap regularization suppresses the universal blue
stripe/high-frequency collapse mode. It is not yet a final success because the
data-only full-mask/full-video objective still shows texture and color
attractors in qualitative side-by-side videos.

H20 `exp6_d2_nocomp_k4_wingap_lose025_beta10_s1s2_4000` is running and must
continue. Action: monitor only; do not kill. Purpose: no-comp data-only
comparison against Exp5 comp.

2026-06-02 H20 check:

```text
status = running
current_stage = Stage2
stage1_completed = 2026-06-01 22:51 CST
stage2_progress = about 420 / 4000 steps
num_gpus = 6
gpu_policy = using GPU 0-5 only; GPU 6/7 idle
stage2_handoff = loaded Stage1 last_weights
qual30 = not started
full_vbench = not started
```

Current H20 diagnostics remain mixed: winner anchoring keeps `win_gap` near
zero and `mse_w_over_ref_mse_w` close to 1, but `loser_dominant_ratio=1.0`,
`mse_l_over_ref_mse_l` is high, and `sigma_term` can approach saturation. This
is still a valid no-comp comparison run; continue monitoring and do not stop it
unless explicitly requested.

Next gate:

```text
EXP_NAME = exp7_d2_comp_k4_partial_wingap_lose025_beta10_s1s2_gate1500
status = planned / launching on PAI
purpose = test whether aligning the training task with D2 partial-mask loser
          generation stabilizes training
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
stage1_steps = 1500
stage2_steps = 1500
validation = qual30 side-by-side + dpo_diag summary
full_vbench = disabled by default for gate
```

Exp5/6 remain data-only full-mask bridge experiments. Exp7 is a task-alignment
gate: DiffuEraser receives the same partial mask used to generate the D2 loser.
If Exp7 is more stable than Exp5, D2 is not inherently bad; the main issue is
the data-only full-mask objective mismatch.

## 2026-06-01 Exp7 Gate1500 Evaluation Status

`exp7_d2_comp_k4_partial_wingap_lose025_beta10_s1s2_gate1500` completed the
planned gate training:

```text
train_mask_mode = partial
mask_from_manifest = true
beta_dpo = 10
winner_abs_reg_weight = 0.05
winner_gap_reg_weight = 1.0
lose_gap_weight = 0.25
stage1_steps = 1500
stage2_steps = 1500
```

Observed with the current full-mask-style qual30:

- Full-mask qual30 looks poor and stripe-heavy.
- Some samples are worse than the new Exp5 winner-anchored full-mask run.
- Winner-gap regularization keeps `win_gap` relatively bounded.
- Loser degradation remains strong: `loser_dominant_ratio` reaches 1.0 and
  `mse_l_over_ref_mse_l` can become very high.

Interpretation:

- Do not mark Exp7 as success.
- Do not mark Exp7 as final failure yet.
- The current qual30 is task-mismatched because Exp7 trains partial-mask
  inpainting (`M_train = M_gen` from manifest `mask_path`), while the qual30
  path is still full-mask prompt generation.
- Current status: **inconclusive / risky**.
- Record full-mask qual30 as **failed / task-mismatched** and run a true
  partial-mask manifest evaluation before deciding whether Exp7 failed.

Completed task-matched eval:

```text
scripts/eval_exp7_partialmask_gate.sh
```

This uses D2 `win_video_path` and `mask_path` to compare DiffuEraser-base and
Exp7 checkpoints on the actual partial-mask inpainting task. Do not launch
full Exp7 4000+4000, full VBench, or Exp8 before this report is reviewed.

## 2026-06-02 Exp7-PM-Gate1500 Partial-Mask Eval

Name:

```text
Exp7-PM-Gate1500 = D2-comp partial-mask task-alignment gate
```

Output:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/partialmask_eval/exp7_gate1500_20260602_000500
```

Artifacts:

```text
side_by_side videos = 60
index.html = present
pair_manifest.csv = present
metrics/summary.csv = present
report.md = present
```

The PAI eval initially failed while reading generated mp4 files because
`imageio` selected a `pyav` backend incompatible with the installed `av`
version (`ContainerFormat.variable_fps` missing). The checked-in eval tool now
uses ffmpeg rawvideo decoding for input videos.

Result:

| Model | mask_region_psnr_mean | mask_region_ssim_mean | outside_region_diff_mean_mean | temporal_diff_delta_vs_gt_mean |
| --- | ---: | ---: | ---: | ---: |
| DiffuEraser-base | 8.99765 | 0.272146 | 2.91477 | 5.58378 |
| Stage1_last | 9.57079 | 0.288404 | 2.92006 | 12.7824 |
| Stage2_last | 7.88448 | 0.235938 | 2.91600 | 6.52143 |

Interpretation:

- Full-mask qual30 remains failed / task-mismatched for Exp7.
- True partial-mask eval shows `Stage1_last` beats DiffuEraser-base on
  mask-region PSNR and SSIM.
- `Stage2_last` regresses below both `Stage1_last` and DiffuEraser-base.
- Stage1 checkpoint-500 and checkpoint-1000 were not available in the exported
  run directory, so only Stage1 last and Stage2 last were evaluated.
- Exp7 is no longer a total failure: partial-mask task alignment is promising.
- Do not launch full Exp7 4000+4000 yet; Stage2 regression must be addressed.
- The prepared no-lose-gap gate is the next likely diagnostic if visual review
  confirms loser-degradation artifacts.

## 2026-06-02 Critical Stage Interpretation Correction

DiffuEraser is a two-stage model:

- Stage1 mainly controls spatial generation quality, BrushNet, UNet2D, and
  appearance.
- Stage2 mainly controls video temporal consistency, motion module behavior,
  and temporal modeling.

Therefore the Exp7-PM-Gate1500 result must not be interpreted as "final
inference should use Stage1 only." The correct candidate is:

```text
DPO-DiffuEraser-Stage1 spatial / appearance weights
+
frozen SFT-DiffuEraser-Stage2 temporal / motion weights
```

Current conclusion:

- Exp7 Stage1_last improves true partial-mask metrics over DiffuEraser-base.
- Exp7 Stage2_last regresses below both Stage1_last and DiffuEraser-base.
- DPO Stage2 is currently harmful and should remain stopped.
- DPO should target Stage1 spatial quality first.
- The temporal / motion prior should be preserved from a validated SFT Stage2
  checkpoint, ideally the best YouTube-VOS SFT Stage2 if found.

Important checkpoint rule:

- Do not simply load a complete SFT Stage2 checkpoint over the DPO result.
- A complete Stage2 checkpoint may contain both spatial and temporal weights.
- The hybrid must keep DPO Stage1 spatial/appearance weights and only preserve
  SFT Stage2 temporal/motion weights.

New first-priority audit/eval:

```text
name = exp7_pm_dpoS1_sftS2_hybrid_ckptsweep
task = true partial-mask manifest eval
candidate = DPO Stage1 checkpoint X + frozen SFT Stage2 motion checkpoint Y
full_vbench = disabled
training = none
```

Prepared tooling:

- `tools/inspect_diffueraser_stage_weights.py`
- `tools/build_diffueraser_dpoS1_sftS2_hybrid.py`
- `scripts/eval_exp7_dpoS1_sftS2_hybrid_partialmask.sh`
- `scripts/launch_exp7_pm_stage1only_ckptsweep_pai.sh` prepared only; do not
  run automatically.
