# Exp43 H20 MiniMax Stage2 SFT Runner

Status: `H20_EXP43_DATA_READY`

Exp43 is an H20-only MiniMax training-system breakthrough track. It is based on
Exp41 and is explicitly authorized to add an isolated Stage2 SFT ladder runner
under `exp43_h20_minimax_stage2_sft_runner/`.

## Scope

- Solve the Exp41 blocker where existing MiniMax SFT runners hard-cap
  `steps > 10`.
- Add isolated Exp43 code only.
- Run BF16-safe preflight and then gated 30/100/300/500-step experiments only
  when gates unlock them.
- Keep PAI read-only. PAI owns data mining and successful-removal mining;
  H20 owns runner/training-system work.

## Forbidden

- Do not modify MiniMax official source files.
- Do not modify `inference/metrics.py`.
- Do not modify shared trainers.
- Do not rewrite Exp1-Exp42 historical results.
- Do not run PAI GPUs or mutate PAI outputs.
- Do not force-push, merge main, or claim universal adapter/final SOTA.

## Paths

- Branch: `research/exp43-h20-minimax-stage2-sft-runner-20260629`.
- Base branch: `origin/research/exp41-h20-minimax-parallel-bf16-20260629`.
- H20 worktree:
  `/home/nvme01/H20_Video_inpainting_DPO_exp43_minimax_stage2_sft`.
- Local/HAL worktree:
  `/home/hj/H20_Video_inpainting_DPO_exp43_minimax_stage2_sft`.
- Output root:
  `/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp43_h20_minimax_stage2_sft_runner`.
- Log root:
  `/home/nvme01/H20_Video_inpainting_DPO/logs/autoresearch/exp43_h20_minimax_stage2_sft_runner`.
- Runtime root:
  `/home/nvme01/H20_Video_inpainting_DPO/runtime/exp43_h20_minimax_stage2_sft_runner`.

## 2026-06-29 H20 GPU Release Audit

Status: `H20_EXP43_GPU_RELEASE_AUDITED`

H20 GPU0-GPU7 were audited before any Exp43 work. No compute PID was present.
`nvitop` holds `/dev/nvidia*` file handles but is not a compute process. CUDA
smoke passed in the H20 `wan` environment:

- Torch: `2.5.1+cu124`.
- CUDA: `12.4`.
- CUDA available: true.
- GPU count: `8`.
- BF16 supported: true.

No PIDs/PGIDs were killed. No GPU reset, `pkill python`, or `killall python`
was used.

Reports:

- `reports/exp43_h20_gpu_release_audit.md`
- `reports/exp43_h20_gpu_release_audit.csv`

## 2026-06-29 Stage2 SFT Runner Readback

Status: `H20_EXP43_STAGE2_SFT_RUNNER_READBACK_COMPLETED`

Readback confirms:

- Exp41 reached `H20_MINIMAX_SFT_BLOCKED` because all existing MiniMax
  SFT/DPO runners are capped at 10 steps or less.
- This prompt authorizes a new Exp43-isolated runner only under
  `exp43_h20_minimax_stage2_sft_runner/`.
- H20 has the mirrored MiniMax data needed for `train64/search24/shadow24`.
- H20 MiniMax weights resolve under
  `/home/nvme01/H20_Video_inpainting_DPO/weights/minimax_remover/current`.
- Exp41 official protocol audit passed for executable README/test settings:
  `UniPCMultistepScheduler`, `float16`, `num_inference_steps=12`,
  `iterations=6`, raw output primary, no hidden comp, no GT leakage, and no
  mask reversal.
- Exp41 BF16 P0-P7 passed, including DDP8 one-batch training, but Exp43 must
  still run its own preflight and record resolved dtype/backend.
- Exp42 remains PAI-side data mining and has not yet provided a pseudo-success
  pool to H20.
- VOR-Eval is excluded from Exp43 training, selection, and tuning.

Report:

- `reports/exp43_h20_stage2_sft_runner_readback.md`

## 2026-06-29 BF16 Safe Preflight

Status: `H20_EXP43_BF16_SAFE_READY`

Added Exp43-isolated runner files:

- `exp43_h20_minimax_stage2_sft_runner/precision_policy.py`
- `exp43_h20_minimax_stage2_sft_runner/runner_stage2_sft_ladder.py`
- `exp43_h20_minimax_stage2_sft_runner/launch_single_gpu_preflight.sh`
- `exp43_h20_minimax_stage2_sft_runner/launch_ddp_bf16_safe.sh`
- `exp43_h20_minimax_stage2_sft_runner/configs/bf16_safe_preflight.yaml`
- `exp43_h20_minimax_stage2_sft_runner/manifests/exp43_preflight_train_h20.jsonl`
- `exp43_h20_minimax_stage2_sft_runner/tests/test_*.py`

P0-P7 completed on H20:

- P0 torch bf16 matmul/backward: PASS.
- P1 VAE fp32 encode/decode: PASS.
- P2 DiT bf16 forward no grad: PASS.
- P3 DiT bf16 forward/backward with fp32 loss: PASS.
- P4 MiniMax fp32 one-batch SFT + checkpoint reload: PASS.
- P5 MiniMax bf16-safe single-GPU one-batch SFT + checkpoint reload: PASS.
- P6 MiniMax bf16-safe DDP2 one-batch SFT + rank0 checkpoint reload: PASS.
- P7 MiniMax bf16-safe DDP8 one-batch SFT + rank0 checkpoint reload: PASS.

No SIGFPE, OOM, CUDA error, NaN/Inf, or Xid was observed. Final GPU0-GPU7
compute apps were empty. This gate permits data readiness and gated SFT ladder
work, but it does not establish a MiniMax quality-positive claim.

Reports:

- `reports/exp43_h20_bf16_safe_preflight.md`
- `reports/exp43_h20_bf16_safe_preflight.csv`
- `reports/exp43_h20_bf16_safe_preflight_summary.json`

## 2026-06-29 Data Readiness

Status: `H20_EXP43_DATA_READY`

Exp43 Stage2 SFT manifests were built from the H20-safe Exp41/Exp40 LocalDPO v3
minimum pool:

- Train: `64` rows, `64` scene groups, BLENDER/REAL `32/32`.
- Search: `24` rows, `24` scene groups, BLENDER/REAL `12/12`.
- Shadow: `24` rows, `24` scene groups, BLENDER/REAL `12/12`.

Validation:

- Required path failures: `0`.
- Optional path failures: `0`.
- Scene overlap train/search, train/shadow, search/shadow: `0/0/0`.
- VOR-Eval rows: `0`.
- Hard-comp rows: `0`.
- condition/winner/mask/loser frame dirs have at least `17` frames and passed
  first-frame decode.

The full `train96/search32/shadow32` target is not available; Exp43 proceeds
with the minimum-pool caveat.

Reports and manifests:

- `exp43_h20_minimax_stage2_sft_runner/manifests/exp43_stage2_sft_train.jsonl`
- `exp43_h20_minimax_stage2_sft_runner/manifests/exp43_stage2_sft_search.jsonl`
- `exp43_h20_minimax_stage2_sft_runner/manifests/exp43_stage2_sft_shadow.jsonl`
- `reports/exp43_h20_data_readiness.md`
- `reports/exp43_h20_data_readiness.csv`
- `reports/exp43_h20_data_manifest_validation.csv`
- `reports/exp43_h20_data_summary.json`

## Next Gates

1. 30-step SFT gate before any 100-step run.
2. 100-step SFT gate before any 300-step run.
3. DPO and 500-step confirmation only if SFT gates pass.
