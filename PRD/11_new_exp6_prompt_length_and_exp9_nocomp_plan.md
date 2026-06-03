# New Exp6 Prompt-Length Audit And Exp9 Nocomp Plan

status: **active plan**

## Current Global Status

- VideoDPO is the bridge domain.
- YouTube-VOS / DAVIS are the target domains.
- PAI is running or preparing Exp9 D3-comp Stage1 gate.
- H20 GPUs 0-5 are available for complementary work.

H20 will be used for:

- new Exp6 prompt-length stratified audit.
- Exp9 D3-nocomp Stage1 gate.

## Metric Policy

| task | metric backend |
| --- | --- |
| video generation / full-mask prompt generation | VBench |
| video inpainting / partial-mask inpainting | `inference/metrics.py` via wrapper |

Do not reimplement PSNR, SSIM, LPIPS, or Ewarp.

## New Exp6 Observation

Observed qualitatively by human review: new Exp6 no-comp may look better than
DiffuEraser-base on longer prompts in the full-mask qual30 side-by-side.

Status: **hypothesis only**.

Required audit:

```text
tools/analyze_new_exp6_prompt_length_effect.py
```

Outputs:

```text
logs/analysis/new_exp6_prompt_length/
reports/new_exp6_prompt_length_audit.md
```

The audit must stratify prompts by character length and generate contact sheets.
Do not turn the long-prompt observation into a conclusion without labels or a
larger prompt set.

## Exp9 D3 Comp Vs Nocomp

PAI:

- `exp9_youtubevos_d3_partialmask_wingap_lose025_stage1_gate1500`
- data: D3 selected-primary-comp

H20:

- `exp9_youtubevos_d3_nocomp_partialmask_wingap_lose025_stage1_gate1500_h20`
- data: D3 selected-primary-nocomp
- GPUs: 0-5
- Stage1 only
- no DPO Stage2
- no VBench for inpainting eval

## 2026-06-03 Monitor Snapshot

New Exp6 prompt-length audit:

- The prompt-length contact sheets were generated under
  `logs/analysis/new_exp6_prompt_length/`.
- Human visual review supports the hypothesis that new Exp6 can beat
  DiffuEraser-base on several longer prompts, but the result is still a
  bridge-domain qualitative pattern rather than a final target-domain metric
  conclusion.

PAI Exp9-comp:

```text
status = manually stopped after overshooting gate; invalid as Exp9 gate
stop_report = reports/pai_exp9_comp_gate_stop_report.md
expected_step_limit = 1500
observed_step = about 4856 / 10000
checkpoint_status = checkpoint-2000 and checkpoint-4000 under stale Exp5-named output dir
stale_output_dir = /mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260603_065327_exp5_d2_comp_k4_stage2_full
```

H20 Exp9-nocomp:

```text
status = finished normally
report = /home/nvme01/H20_Video_inpainting_DPO/reports/h20_exp9_nocomp_gate_monitor_report.md
current_step = 1500 / 1500
max_steps_detected = 1500
checkpoint_status = checkpoint-500, checkpoint-1000, checkpoint-1500, last_weights
gpu_policy = launched on GPU 0-5 only; GPUs idle after completion
```

H20 nocomp is ready for target-domain inpainting evaluation. PAI comp must be
rerun with the stale-env-safe launcher before a fair comp-vs-nocomp comparison.

2026-06-04 CST update:

- PAI clean Exp9-comp is running with the stale-env-safe launcher and confirmed
  `max_steps=1500`, `ckpt_steps=500`, `ckpt_limit=5`.
- H20 Exp9-nocomp finished the Stage1 gate and has `checkpoint-500`,
  `checkpoint-1000`, `checkpoint-1500`, and `last_weights`.
- H20 target eval has been launched for the D3 selected-primary-nocomp
  YouTube-VOS-derived manifest:
  `logs/target_eval/exp9_d3_nocomp_gate_h20_20260604_022414`.
- DAVIS target eval remains blocked until prediction videos and a validated
  `pair_manifest` are produced.

## Decision Matrix

| result | next step |
| --- | --- |
| PAI Exp9-comp > base and H20 Exp9-nocomp <= base | use comp for target-domain DPO |
| H20 Exp9-nocomp > base and PAI Exp9-comp <= base | use no-comp for target-domain DPO |
| both improve | compare metric and qualitative stability, then consider Stage1 sweep to 3000 |
| both fail | stop direct DPO; consider target-domain SFT warmup or no-lose gate |
| long-prompt Exp6 effect appears prompt-dependent | document as bridge-domain prompt-conditioning effect only |

## Do Not Do

- Do not stop PAI Exp9.
- Do not use VBench for inpainting.
- Do not train DPO Stage2.
- Do not start Exp8.
- Do not regenerate D2 or D3.
- Do not use broken manifest paths.
