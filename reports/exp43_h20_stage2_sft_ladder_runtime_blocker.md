# Exp43 H20 Stage2 SFT Ladder Runtime Blocker

Status: `H20_EXP43_SFT_BLOCKED`

Branch: `research/exp43-h20-minimax-stage2-sft-runner-20260629`  
Commit: `eda52a8757b2354e710bea317e0fd07130436d4c`

## Completed Cell

Run: `SFT-A_lr3em5_step30`

- Recipe: `SFT-A`
- LR: `3e-5`
- Target steps: `30`
- World size: `8`
- Precision: `bf16_safe`
- Training status: `TRAIN_DONE`
- Peak rank0 VRAM: `66414.96484375` MiB
- Checkpoint:
  `/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp43_h20_minimax_stage2_sft_runner/sft_ladder/SFT-A_lr3em5_step30/checkpoints/checkpoint-30`
- SIGFPE/OOM/CUDA/NaN/Inf: none observed

This confirms that Exp43 solved the `steps > 10` runner blocker for one true
30-step 8GPU MiniMax SFT cell. It does not establish a quality-positive result.

## Evaluation Result

Evaluation completed on `search24` and `shadow24`; `48` metric rows and `48`
visual-review rows were generated.

Search aggregate deltas, Step30 minus Step0:

- full PSNR: `-5.833128230661999`
- mask PSNR: `-4.674487775418862`
- boundary PSNR: `-4.700944147600658`
- outside PSNR: `-7.594088453813615`
- Ewarp: `+0.6460841968352801`
- temporal diff MAE: `+0.9358525502805909`
- full PSNR win rate: `0.0`
- mask PSNR win rate: `0.08333333333333333`

Shadow aggregate deltas, Step30 minus Step0:

- full PSNR: `-6.55060498000691`
- mask PSNR: `-4.223185495799285`
- boundary PSNR: `-5.373455771430662`
- outside PSNR: `-8.45318893655187`
- Ewarp: `+0.5934015673112469`
- temporal diff MAE: `+0.853355257306248`
- full PSNR win rate: `0.041666666666666664`
- mask PSNR win rate: `0.0`

Gate result: `FAIL`.

Blockers:

- Search and shadow PSNR/local-region metrics are strongly negative.
- Ewarp worsens above the gate tolerance.
- Visual review is still pending; no metric-only promotion is allowed.
- LPIPS remains runtime-blocked, described below.

No 100-step, 300-step, DPO-after-SFT, or 500-step confirmation was unlocked.

## LPIPS Blocker

Initial evaluation produced `LPIPS = NaN` because H20 `wan` did not include
`torchmetrics`.

Mitigation attempted:

- Installed `torchmetrics==1.4.0`, `lightning-utilities==0.11.7`,
  `pretty-errors==1.2.25`, and `colorama==0.4.6` into the Exp43 runtime-local
  dependency directory:
  `/home/nvme01/H20_Video_inpainting_DPO/runtime/exp43_h20_minimax_stage2_sft_runner/python_deps`
- Did not install or replace Torch/CUDA packages.
- `torchmetrics` LPIPS import smoke passed using `PYTHONPATH`.
- Reran `evaluate-sft --reuse-existing` for the same checkpoint.

The LPIPS rerun was stopped after exceeding reasonable runtime:

- tmux session: `exp43_sft_a_30_lpips`
- owned PGID: `3675531`
- elapsed when stopped: approximately `01:26:30`
- Ewarp progress messages observed: `56`
- Result: metrics remained at old `LPIPS BLOCKED` values

Only the Exp43 LPIPS rerun process group was terminated. No external process
was killed.

## GPU State After Stop

After stopping the Exp43 LPIPS rerun:

- Exp43 GPU0 process: released
- GPU0-GPU6: no Exp43 compute process
- GPU7: occupied by an external non-Exp43 process:
  `/home/nvme03/workspace/lingbot-world/.conda_envs/lingbot-world-v2/bin/python`
  with PID `1250053`, around `63540` MiB at the last audit

The external GPU7 process was not touched. Because Exp43 requires 8GPU DDP for
the ladder, remaining 30-step cells were not launched.

## Claim Boundary

- `H20_EXP43_SFT_30STEP_PASS`: not claimed
- `H20_EXP43_DPO_100STEP_PASS`: not claimed
- `H20_MINIMAX_THIRD_BACKBONE_POSITIVE`: not claimed
- Universal adapter / final SOTA / top-conference novelty: not claimed

Current scientific conclusion: MiniMax Stage2 SFT is not positive on the one
completed 30-step cell; the broader ladder is blocked by LPIPS runtime behavior
and loss of exclusive 8GPU availability.
