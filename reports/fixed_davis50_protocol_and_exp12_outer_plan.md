# Fixed DAVIS50 Protocol And Exp12 Outer Boundary Plan

## Canonical Evaluation Protocol

All current and future target-domain inpainting evaluation for Exp9+ must use:

- DAVIS50
- raw6 (`NUM_INFERENCE_STEPS=6`)
- hard comp
- D+G off:
  - no mask dilation
  - no Gaussian blur during comp
- no PCM
- frame-wise in-memory metric
- metric wrapper: `tools/run_davis50_framewise_protocol_eval.py`

Do not use the generic saved-output pair-manifest metric path for final tables,
because it can drift from the frame-wise hard-comp protocol and produced lower
non-comparable scores.

## Current Fixed Baseline

The latest canonical rerun under
`20260614_145632_exp11_exp12_framewise_raw6_davis50_tcfix` recovered the expected
32+ PSNR regime:

- SFT48000 baseline PSNR: `32.731391`
- Best current result: `Exp11_boundary_outer_b075_S2`, PSNR `33.013954`

## Original Exp12 Boundary Setting

Original `exp12_adaptive_normalization` used:

- `boundary_mode=exp10_default`
- `boundary_weight=0.5`
- `adaptive_norm_mode=batch_zscore`

Therefore original Exp12 was not testing the best Exp11 boundary setting.

## New Isolated Branch

Created isolated branch:

- folder: `exp12_adaptive_outer_boundary/`
- registry: `experiment_registry/exp12_adaptive_outer_boundary/`
- launcher: `scripts/launch_exp12_adaptive_outer_boundary_pai.sh`
- variant: `exp12_batch_adaptive_outer_b075_s1s2_2000`
- boundary setting: `outer`, `boundary_weight=0.75`
- normalization: `batch_zscore` after log-ratio

Decision rule:

- If `Exp12 adaptive + outer b0.75` beats `Exp11 outer b0.75 S2`, keep it as the next candidate.
- If not, keep the original normalization result as an ablation and keep `Exp11 outer b0.75 S2` as current best.

## Exp12 Adaptive + Outer b0.75 Result

PAI run:

- stage1: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260615_043419_exp12_adaptive_outer_exp12_batch_adaptive_outer_b075_s1s2_2000_s1_2000_davis_pai`
- stage2: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260615_043419_exp12_adaptive_outer_exp12_batch_adaptive_outer_b075_s1s2_2000_s2_2000_davis_pai`
- stage1 eval: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp12_batch_adaptive_outer_b075_s1s2_2000_stage1_val_davis_20260615_043419_exp12_adaptive_outer/metrics/summary.csv`
- stage2 eval: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp12_batch_adaptive_outer_b075_s1s2_2000_stage2_val_davis_20260615_043419_exp12_adaptive_outer/metrics/summary.csv`

| Method | Stage | PSNR | SSIM | LPIPS | VFID | TC | Mask PSNR |
|---|---|---:|---:|---:|---:|---:|---:|
| Exp12 adaptive outer b0.75 | DPO-S1 + SFT-S2 | 32.847530 | 0.971693 | 0.015612 | 0.184848 | 0.971164 | 24.001063 |
| Exp12 adaptive outer b0.75 | DPO-S1 + DPO-S2 | 32.856975 | 0.971585 | 0.015605 | 0.193578 | 0.971475 | 24.010508 |
| Exp11 boundary outer b0.75 | DPO-S1 + DPO-S2 | 33.013954 | 0.972295 | 0.015363 | 0.175423 | 0.971122 | 24.167487 |

Conclusion:

- Exp12 adaptive + outer b0.75 did **not** beat `Exp11_boundary_outer_b075_S2`.
- Adaptive normalization with the best current outer boundary setting should be kept as an ablation, not the current best method.
- Current best remains `Exp11_boundary_outer_b075_S2` under the fixed DAVIS50 raw6 hard-comp protocol.
