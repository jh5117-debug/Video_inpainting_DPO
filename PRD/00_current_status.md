## 2026-06-06 Exp8a Full-Loss Regularized DPO DAVIS Continuation

Current PAI status is based on the user's pasted audit output. PAI remains manual-only for Codex: do not claim direct execution on PAI.

Exp8 has been split into two evidence units:

- `Exp8a`: D3 comp + ordinary full-loss regularized DPO. This is the active baseline now.
- `Exp8b`: future region-weighted-loss ablation. Do not conflate it with the current run.

Current Exp8a definition:

```text
experiment = exp08_d3_comp_fullloss_wingap_lose025_s1s2_2000_davis_pai
data = D3 selected_primary_comp.repaired.pai_paths.jsonl
task = partial-mask video inpainting
weights = /mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000
prior = ProPainter prior required for DAVIS validation
loss_region_mode = full
stage = Stage1 2000 -> Stage1 DAVIS val -> Stage2 2000 -> Stage2 DAVIS val
metric = tools/run_inpainting_metric_eval.py / inference/metrics.py
VBench = not used
```

Exp8a true loss:

```text
m_w     = policy winner MSE
m_l     = policy loser MSE
m_w_ref = reference winner MSE
m_l_ref = reference loser MSE
win_gap  = m_w - m_w_ref
lose_gap = m_l - m_l_ref

L_total =
    -logσ{-0.5 * 10 * (win_gap - 0.25 * lose_gap)}
    + 0.05 * m_w
    + ReLU(win_gap)
```

Current observed status:

- Stage1 training completed on PAI at 2000/2000.
- Stage1 run dir:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260605_142442_exp08_d3_comp_fullloss_wingap_lose025_s1_2000_davis_pai`
- Confirmed Stage1 artifacts: `checkpoint-2000`, `last_weights`, `dpo_diagnostics.csv`.
- Latest continuation log:
  `logs/pipelines/exp08_d3_comp_fullloss_continue_after_s1_fixsafety_len24_pai_20260606_054617.log`
- Latest continuation PID at pasted audit: `809695`.
- Current phase at latest audit: Stage1 DAVIS validation, DiffuEraser-base inference producing 4-in-1 videos.
- Current Stage1 validation output:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp08a_fullloss_stage1_val_davis_20260605_142442_continue_fixsafety_len24_20260606_054617`
- Not yet confirmed: `DPO-S1_SFT-S2` inference, `metrics/summary.csv`, Stage2 start, or Stage2 validation.

Fixes already applied on PAI during continuation:

- Added missing SD1.5 `feature_extractor/preprocessor_config.json`.
- Disabled missing SD1.5 safety checker in `diffueraser/diffueraser.py` by passing `safety_checker=None`, `feature_extractor=None`, and `requires_safety_checker=False`.
- Relaunched DAVIS validation with `DAVIS_VIDEO_LENGTH=24` after the 16-frame run failed because the effective duration was below the DiffuEraser/ProPainter minimum.

Do not rerun Exp8a Stage1. Next action is only to monitor the active continuation and wait for Stage1 validation, Stage2 training, Stage2 validation, metrics, side-by-side videos, and dpo summaries.

## 2026-06-05 Exp8 DAVIS Region-Loss Diagnostic Prepared

Current execution boundary:

- H20 is generating new Exp7-fix small-mask D2 data. Do not stop H20, kill jobs, start H20 training, or take H20 GPUs.
- PAI is manual-only from Codex. The agent may prepare scripts and a copy/paste command block, but must not claim PAI execution.

Prepared next PAI diagnostic:

```text
experiment = exp08_d3_comp_regionloss_wingap_lose025_s1s2_2000_davis_pai
registry = experiment_registry/exp08_d3_comp_regionloss_davis_stage1stage2_2000
launcher = scripts/launch_exp8_d3_comp_regionloss_s1s2_2000_davis_pai.sh
data = D3 selected_primary_comp.repaired.pai_paths.jsonl
task = partial-mask video inpainting
weights = /mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000
prior = ProPainter prior required for DAVIS inference/validation
loss_region_mode = region
stage = Stage1 2000 -> DAVIS val -> Stage2 2000 -> DAVIS val
metric = tools/run_inpainting_metric_eval.py using inference/metrics.py
VBench = not used
```

Exp8 true loss:

```text
m_w, m_l, m_w_ref, m_l_ref are region-weighted MSE values.
win_gap = m_w - m_w_ref
lose_gap = m_l - m_l_ref

L_total =
    -logσ{-0.5 * 10 * (win_gap - 0.25 * lose_gap)}
    + 0.05 * m_w
    + ReLU(win_gap)

region weights: mask=1.0, boundary=0.5, outside=0.05
```

The launcher must stop before training if the D3 PAI manifest, SFT-48000 weights, ProPainter weights, DAVIS data, region-loss implementation, or dpo diagnostics support is missing.

## 2026-06-04 PAI Audit Backfill / Registry Correction

This update uses the returned PAI audit archive `pai_experiment_registry_reports.tar.gz` and `reports/pai_experiment_registry_paths.csv`.

Key corrections:

- Exp5 is split into three evidence units: Old Exp5 beta500 plain collapse, Exp5 beta10 plain collapse, and New Exp5 winner-gap/lose025 improvement.
- New Exp6 remains D2 no-comp + the New Exp5 winner-preserving loss. It is not plain Exp6.
- PAI audit found remote `dpo_diagnostics.csv` paths for Old Exp5, Exp5 beta10 plain, New Exp5, Exp7, Exp8, and Exp9-comp. These are marked `REMOTE_DIAG_PATH_FOUND` until the actual CSVs are fetched or summarized on PAI.
- Local H20 diagnostic snapshots exist for New Exp6, Exp9 D3-nocomp, and Exp9 D3 no-lose.
- Exp4 still has no artifact path in the audit; it remains a deleted/failed fullmask quality gate.
- YouTube-VOS/D3 work must use SFT-48000 DiffuEraser, not a naked base. H20 confirmed path: `/home/nvme01/H20_Video_inpainting_DPO/weights/diffuEraser/converted_weights_step48000`; PAI required path: `/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000`.
- Partial-mask inpainting should use ProPainter prior; fullmask/video-generation bridge runs should not rely on that prior.
- Exp7 must be repaired with 15%-20% masks and ProPainter prior before expanding D3/YouTube-VOS DPO.

Evidence files:

- `experiment_registry/experiment_matrix.md`
- `reports/pai_experiment_registry_audit_summary.md`
- `reports/all_experiments_dpo_diag_summary.md`
- `reports/exp7_prior_mask_audit.md`

## 2026-06-04 Exp7 Small-Mask / Prior Reset

This update supersedes any plan to expand D3/YouTube-VOS DPO before Exp7 is fixed.

Required facts:

- YouTube-VOS / D3 work must use the fine-tuned SFT-48000 DiffuEraser weights, not an ordinary base.
  - PAI path: `/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000`
  - HAL local reference: `/home/hj/Video_inpainting_DPO/weights/diffuEraser/converted_weights_step48000`
  - H20 confirmed path: `/home/nvme01/H20_Video_inpainting_DPO/weights/diffuEraser/converted_weights_step48000`
- Every experiment must have an `experiment_registry/<exp>/` folder with config, commands, paths, status, metric, qualitative, and dpo_diag notes.
- Every DPO experiment needs `dpo_diagnostics.csv`; missing diagnostics are marked `MISSING_DPO_DIAG`.
- Current lineage is: Exp4 failed fullmask data gate; Old Exp5 plain DPO collapse; New Exp5 comp + winner anchoring improves but is not final; New Exp6 no-comp + same winner anchoring is not plain Exp6; Exp7 changed task to partial mask but is suspicious; Exp8/Exp9 are target-domain diagnostics only.
- Fullmask/video-generation settings should not rely on ProPainter prior. Partial-mask video inpainting should use DiffuEraser/OR ProPainter prior.
- VideoDPO partial-mask masks should be reduced to 15%-20% for Exp7-fix; do not continue old large-mask D2 for new Exp7 gates.
- Train may use YouTube-VOS, but final target eval should use DAVIS at `/mnt/workspace/hj/nas_hj/data/external/davis_432_240`.

Current Exp7 audit:

- `tools/eval_generated_loser_partialmask_model.py` appears to use a direct pipeline path with masked winner frames and masks, without `inference/run_OR.py` or `--propainter_model_dir`.
- This makes the current Exp7 eval likely no-prior and may explain why SFT/base videos also look bad.
- Report: `reports/exp7_prior_mask_audit.md`.

Next planned gates after small-mask/prior data exists:

- PAI: `exp07_fix_smallmask_prior_wingap_lose025_stage1_gate1500`.
- H20: `exp07_fix_smallmask_prior_wingap_nolose_stage1_gate1000`.

Do not run DPO Stage2, VBench for inpainting, or long D3 sweeps until this sanity gate is reviewed.

# Current Status

## 2026-06-04 Experiment Registry / DPO-Diag Audit Update

A formal `experiment_registry/` has been added as the source-of-truth artifact ledger. Each experiment now has an independent folder with README, paths, commands, dpo_diag, metrics, qualitative, and status notes.

Naming corrections:

- Old Exp5 and New Exp5 are separate rows and must not be merged.
- New Exp6 is the D2 no-comp + winner-anchored-loss diagnostic, not a plain Exp6.
- Exp4 is a failed full-mask generated-loser quality gate / diagnostic-only artifact.
- Exp8 is a target-domain D3 region-loss gate; it is not a completed VideoDPO bridge success claim.
- Exp9 is the YouTube-VOS/D3 target-domain partial-mask gate family.

Evidence rule:

- Every DPO experiment must have `dpo_diagnostics.csv` or be explicitly marked `MISSING_DIAG`.
- Current conclusions must be supported by metric + qualitative + dpo_diag together, or marked as incomplete / diagnostic-only.
- The aggregate dpo summary is generated by `tools/collect_dpo_diag_summaries.py` and written to `reports/all_experiments_dpo_diag_summary.md`.

Current audit status:

- H20 dpo_diag snapshots found: New Exp6 Stage1/Stage2, Exp9 D3-nocomp Stage1, Exp9 D3-comp no-lose Stage1.
- Missing / pending PAI dpo_diag: historical Exp1-3, Exp4, Old Exp5, New Exp5, Exp7, Exp7 hybrid, Exp8, Exp9 D3-comp clean gate.
- PAI must be scanned manually with the fixed-glob audit command; do not recursively scan the whole NAS.


Updated: 2026-06-02

## 2026-06-04 CST Exp9 Clean Gate / Target Eval Update

PAI clean Exp9-comp gate:

```text
experiment = exp9_youtubevos_d3_partialmask_wingap_lose025_stage1_gate1500
status = clean gate running
manual_launch_report = reports/pai_exp9_comp_clean_launch_report.md
manifest = selected_primary_comp.repaired.pai_paths.jsonl
config_confirmed = Stage1 only, partial mask, mask_from_manifest=true, beta=10,
                   winner_gap=1.0, lose_gap=0.25, max_steps=1500,
                   ckpt_steps=500, ckpt_limit=5
invalid_old_run = do not evaluate
```

H20 Exp9-nocomp gate:

```text
experiment = exp9_youtubevos_d3_nocomp_partialmask_wingap_lose025_stage1_gate1500_h20
status = Stage1 gate completed
checkpoints = checkpoint-500, checkpoint-1000, checkpoint-1500, last_weights
directly_evaluable_now = last_weights
note = checkpoint-500/1000/1500 are accelerator state dirs and need export
       before direct inference eval
```

H20 target-domain eval status:

```text
youtubevos_d3_nocomp_eval = completed on H20 using D3 selected-primary-nocomp
eval_log = logs/pipelines/exp9_d3_nocomp_target_eval_20260604_023243.log
eval_output = logs/target_eval/exp9_d3_nocomp_gate_h20_20260604_023243
samples = 200 generated videos, 30 side-by-side videos, 100 metric samples
metric_summary = Exp9_nocomp_last improves mask PSNR/SSIM over base but worsens
                 whole-video SSIM, boundary PSNR, outside max diff, and temporal
                 delta; do not continue nocomp long training before comp eval.
davis_eval = blocked until target prediction generation and pair_manifest are prepared
metric_policy = use inference/metrics.py through project wrappers; no VBench
```

## 2026-06-03 Exp9 Gate Monitoring Boundary

PAI and H20 execution boundaries are explicit:

- PAI is manual-only from Codex. Provide copy/paste shell blocks and wait for
  pasted output before judging or stopping a run.
- H20 can be inspected by SSH from HAL. H20 monitoring reports should be
  written directly under `/home/nvme01/H20_Video_inpainting_DPO/reports/`.

Current Exp9 target-domain gates:

| Host | Experiment | Status | Manifest | Max steps | Action |
| --- | --- | --- | --- | ---: | --- |
| PAI | `exp9_youtubevos_d3_partialmask_wingap_lose025_stage1_gate1500` | stopped; invalid as Exp9 gate | D3 selected-primary-comp PAI paths | actual log showed 10000 | do not use as Exp9 gate; stale env wrote old Exp5 run name |
| H20 | `exp9_youtubevos_d3_nocomp_partialmask_wingap_lose025_stage1_gate1500_h20` | finished normally | D3 selected-primary-nocomp H20 paths | 1500 confirmed | ready for target-domain eval |

H20 monitor snapshot, 2026-06-03 14:21 CST:

```text
report = /home/nvme01/H20_Video_inpainting_DPO/reports/h20_exp9_nocomp_gate_monitor_report.md
status = running
current_step = about 190 / 1500
max_steps_detected = 1500
stage_policy = Stage1 only; no DPO Stage2; no VBench
gpu_policy = GPU 0-5 only; GPU 6/7 idle in the initial launch check
checkpoint_status = no checkpoint yet; wait for checkpoint-500 / 1000 / 1500
dpo_diagnostics_csv = present
errors = no Traceback / OOM / SIGFPE detected in monitor scan
```

PAI monitor / stop update, 2026-06-03 15:03 CST:

```text
report = reports/pai_exp9_comp_gate_stop_report.md
status = manually stopped
evidence = global_step about 4850 and progress 4856 / 10000
stopped_pid = 3400017
final_process_check = empty for exp9/train_stage1/accelerate/lingbot-worldphy matcher
checkpoint_status = saved under stale output dir, not an Exp9-named gate dir
stale_output_dir = /mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260603_065327_exp5_d2_comp_k4_stage2_full
available_checkpoints = checkpoint-2000, checkpoint-4000
interpretation = invalid / contaminated by stale env; not comparable to gate1500
```

Updated H20 monitor snapshot, 2026-06-04 01:08 CST:

```text
status = finished normally
current_step = 1500 / 1500
max_steps_detected = 1500
gpu_policy = all GPUs idle after completion
checkpoint_status = checkpoint-500, checkpoint-1000, checkpoint-1500, last_weights
errors = no Traceback / OOM / SIGFPE detected
```

Next after both gates have evaluable checkpoints:

- Run target-domain video inpainting evaluation on YouTube-VOS / DAVIS.
- Use `tools/run_inpainting_metric_eval.py`, which calls
  `inference/metrics.py`.
- Do not use VBench for Exp9 inpainting evaluation.
- Do not train DPO Stage2, start Exp8, regenerate D2/D3, or launch a long run
  before the comp-vs-nocomp gate report is reviewed.

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
## 2026-06-02 Mainline Correction: Target Domain

The final evaluation and demo domains are **YouTube-VOS** and **DAVIS**.
VideoDPO is a bridge domain only: it is useful for native VideoDPO repo
integration, DiffuEraser adapter validation, generated-loser manifest plumbing,
partial-mask support, DPO diagnostics, and Stage1/Stage2 loading ablations. It
is not the final target domain.

Do not continue planning VideoDPO partial-mask SFT warmup. If SFT is needed, it
should be target-domain SFT or related video-inpainting data, not VideoDPO
bridge warmup.

Current priority:

1. Run target-domain eval on YouTube-VOS / DAVIS using existing checkpoints and
   stage compositions.
2. Sync D3 YouTube-VOS generated-loser data to PAI only as background data
   preparation.
3. After D3 sync, run D3 post-generation audit, path rewrite, and readiness.
4. Enter Exp9 target-domain DPO only if target-domain eval shows that
   VideoDPO-bridge DPO does not transfer.

H20 D3 audit:

- root: `/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4`
- report: `/home/nvme01/H20_Video_inpainting_DPO/reports/d3_h20_audit_report.md`
- size/files: 249G / 1,819,879 files
- selected primary rows: 3,327 comp and 3,327 no-comp
- sampled 100 rows: status OK, 16 frames, 512x320, readable
- all sampled paths are H20-only `/home/nvme01/...`; PAI path rewrite is required before training.

## 2026-06-03 Target-Domain Mainline

Current mainline:

- VideoDPO is a bridge domain only. It validates model replacement, native
  VideoDPO integration, generated-loser manifests, partial-mask plumbing,
  winner-gap regularized DPO, diagnostics, and Stage1/Stage2 loading.
- YouTube-VOS and DAVIS are the final target domains for evaluation and
  reporting.
- Do not start Exp8, VideoDPO warmup, DPO Stage2, full VBench, or any long
  4000+4000 run in this phase.

Metric policy:

| Task type | Metric backend |
| --- | --- |
| video generation / full-mask VBench prompt generation | VBench |
| video inpainting / partial-mask inpainting | project metric module (`inference/metrics.py`; no standalone `metric.py` exists in this checkout) |

The target-domain YouTube-VOS / DAVIS partial-mask eval must use the existing
project metric functions through `tools/run_inpainting_metric_eval.py`. Do not
reimplement PSNR, SSIM, mask-region metrics, boundary metrics, or temporal
metrics in new scripts. New wrappers may only organize inputs, call the
existing metric module, and summarize results.

D3 status:

- D3 is the YouTube-VOS generated-loser data asset for Exp9.
- PAI slim sync completed for selected-primary data.
- D3 full readiness = false under the current slim sync because secondary
  manifests are absent.
- D3 primary-comp gate readiness = true once the selected-primary-comp repaired
  manifest has clean PAI paths and the primary-comp sample audit has zero
  issues. Secondary absence must not block the first Exp9 Stage1 gate.
- `selected_primary_comp.repaired.pai_paths.jsonl` is the first Exp9 gate
  training manifest if readiness checks pass on PAI.
- Slim sync does not make candidates/secondary manifests fully ready; do not
  run workflows that require full D3 candidates or secondary selections unless
  they are synced separately.

Exp9 boundary:

- Exp9 is target-domain partial-mask DPO.
- First run is Stage1-only gate:
  `exp9_youtubevos_d3_partialmask_wingap_lose025_stage1_gate1500`.
- Train with `train_mask_mode=partial`, `mask_from_manifest=true`,
  `loss_region_mode=full`, winner anchoring, `beta_dpo=10`,
  `lose_gap_weight=0.25`, `winner_abs_reg_weight=0.05`,
  `winner_gap_reg_weight=1.0`, and `sft_reg_weight=0.0`.
- Do not train DPO Stage2. Eval uses target-domain inpainting metrics, not
  VBench.

## 2026-06-03 H20 Complementary Work

Current split:

- PAI: Exp9 D3-comp Stage1 gate.
- H20: Exp9 D3-nocomp Stage1 gate on GPUs 0-5.

Reason:

- New Exp6 no-comp full-mask qual30 has a qualitative hypothesis that longer
  prompts may look better than DiffuEraser-base.
- This observation is not a conclusion; it requires prompt-length stratified
  audit via `tools/analyze_new_exp6_prompt_length_effect.py`.
- Since PAI is running the comp target-domain gate, H20 should run the nocomp
  target-domain gate to form a comp-vs-nocomp comparison.

H20 nocomp gate:

- `exp9_youtubevos_d3_nocomp_partialmask_wingap_lose025_stage1_gate1500_h20`
- manifest: D3 `selected_primary_nocomp.repaired.jsonl` or readable H20-local
  `selected_primary_nocomp.jsonl`
- Stage1 only; no DPO Stage2
- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5`
- target-domain eval uses `inference/metrics.py`, not VBench
