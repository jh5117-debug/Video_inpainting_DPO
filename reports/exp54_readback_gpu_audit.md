# Exp54 PAI VOID SDPO / Linear-DPO Targeted One-Step Readback + GPU Audit

Status: `EXP54_PAI_GPU0_1_READY`

Branch: `research/exp54-void-sdpo-linear-pai-20260701`
HEAD: `a46edb656948799754789f73860a29bf1a469a0c`
Created: `2026-07-01T15:53:57+08:00`

PAI lane only: R3/R4 on GPU0-1.

## Source-of-truth readback

- Exp52 final status: VOID adapter-engineering candidate, not third-backbone evidence.
- Exp52 R1_Q0_T500_S0 was mixed: full/object/boundary/outside improved, affected/overlap regressed.
- 10-step remains locked until Exp55 aggregator sees a one-step PASS.

## GPU audit

- GPU0: 0 MiB used, util 0%, `ready`
- GPU1: 0 MiB used, util 0%, `ready`

## Required command excerpts

```text
git fetch rc=0
From https://github.com/jh5117-debug/Video_inpainting_DPO
 * [new branch]      fix-exp-registry-dpo-diag-ppt -> origin/fix-exp-registry-dpo-diag-ppt
 * [new branch]      main       -> origin/main
 * [new branch]      research/exp20-adaptive-region-autoresearch-20260619 -> origin/research/exp20-adaptive-region-autoresearch-20260619
 * [new branch]      research/exp23-two-stage-pool-morphology-sweep -> origin/research/exp23-two-stage-pool-morphology-sweep
 * [new branch]      research/exp24-multibackbone-dpo-adapter -> origin/research/exp24-multibackbone-dpo-adapter
 * [new branch]      research/exp25-vor-gate16-cli4-20260625 -> origin/research/exp25-vor-gate16-cli4-20260625
 * [new branch]      research/exp25-vor-or-preference-data -> origin/research/exp25-vor-or-preference-data
 * [new branch]      research/exp26-videopainter-dpo-v2 -> origin/research/exp26-videopainter-dpo-v2
 * [new branch]      research/exp27-localdpo-objective-cli4-20260625 -> origin/research/exp27-localdpo-objective-cli4-20260625
 * [new branch]      research/exp27-paper-grounded-preference-study -> origin/research/exp27-paper-grounded-preference-study
 * [new branch]      research/exp28-fine-inner-boundary-sweep-20260625 -> origin/research/exp28-fine-inner-boundary-sweep-20260625
 * [new branch]      research/exp29-minimax-effecterase-adapter-feasibility-20260626 -> origin/research/exp29-minimax-effecterase-adapter-feasibility-20260626
 * [new branch]      research/exp30-vor-or-multimodel-minimax-adapter-20260627 -> origin/research/exp30-vor-or-multimodel-minimax-adapter-20260627
 * [new branch]      research/exp31-videopainter-2000step-longrun-20260627 -> origin/research/exp31-videopainter-2000step-longrun-20260627
 * [new branch]      research/exp32-diffueraser-vor-or-2000step-20260627 -> origin/research/exp32-diffueraser-vor-or-2000step-20260627
 * [new branch]      research/exp33-effecterase-vor-eval-baseline-20260627 -> origin/research/exp33-effecterase-vor-eval-baseline-20260627
 * [new branch]      research/exp34-objective-ablation-prep-postminimax-20260627 -> origin/research/exp34-objective-ablation-prep-postminimax-20260627
 * [new branch]      research/exp35-minimax-flow-dpo-rescue-20260627 -> origin/research/exp35-minimax-flow-dpo-rescue-20260627
 * [new branch]      research/exp36-minimax-objective-rescue-20260627 -> origin/research/exp36-minimax-objective-rescue-20260627
 * [new branch]      research/exp37-minimax-localdpo-badnoise-rescue-20260627 -> origin/research/exp37-minimax-localdpo-badnoise-rescue-20260627
 * [new branch]      research/exp38-minimax-full-adapter-breakthrough-20260628 -> origin/research/exp38-minimax-full-adapter-breakthrough-20260628
 * [new branch]      research/exp39-h20-minimax-mirror-bf16-20260628 -> origin/research/exp39-h20-minimax-mirror-bf16-20260628
 * [new branch]      research/exp40-minimax-psnr-safe-rescue-20260628 -> origin/research/exp40-minimax-psnr-safe-rescue-20260628
 * [new branch]      research/exp41-h20-minimax-parallel-bf16-20260629 -> origin/research/exp41-h20-minimax-parallel-bf16-20260629
 * [new branch]      research/exp42-pai-minimax-successful-removal-badnoise-20260629 -> origin/research/exp42-pai-minimax-successful-removal-badnoise-20260629
 * [new branch]      research/exp43-h20-minimax-stage2-sft-runner-20260629 -> origin/research/exp43-h20-minimax-stage2-sft-runner-20260629
 * [new branch]      research/exp44-pai-minimax-targeted-same-source-mining-20260629 -> origin/research/exp44-pai-minimax-targeted-same-source-mining-20260629
 * [new branch]      research/exp45-h20-diffueraser-videopainter-objective-study-20260629 -> origin/research/exp45-h20-diffueraser-videopainter-objective-study-20260629
 * [new branch]      research/exp45-pai-minimax-pair-scaleup-20260629 -> origin/research/exp45-pai-minimax-pair-scaleup-20260629
 * [new branch]      research/exp46-h20-minimax-exp45-pseudosuccess-sft-20260629 -> origin/research/exp46-h20-minimax-exp45-pseudosuccess-sft-20260629
 * [new branch] 
```

## File read status

- `PRD/00_current_status.md`: present
- `PRD/01_experiment_matrix.md`: present
- `PRD/47_exp50_pai_void_adapter_feasibility.md`: present
- `PRD/48_exp51_void_loser_dominant_rescue.md`: present
- `PRD/49_exp52_void_winner_preserving_allgpu.md`: present
- `experiment_registry/exp50_pai_void_adapter_feasibility/status.md`: present
- `experiment_registry/exp51_void_loser_dominant_rescue/status.md`: present
- `experiment_registry/exp52_void_winner_preserving_allgpu/status.md`: present
- `reports/exp51_void_loser_dominant_forensic.md`: present
- `reports/exp51_void_quadmask_metrics.md`: present
- `reports/exp51_void_quadmask_ablation_data.md`: present
- `reports/exp52_cache_summary.json`: present
- `reports/exp52_r1_row0_smoke.md`: present
- `reports/exp52_rescue_onestep.md`: present
- `reports/exp52_rescue_onestep_summary.json`: present
- `reports/exp52_void_rescue_decision.md`: present
- `reports/exp52_void_next_steps.md`: present
