# Exp43 H20 MiniMax Stage2 SFT Runner

Status: `H20_EXP43_GPU_RELEASE_AUDITED`

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

## Next Gates

1. Exp43 readback and H20 worktree setup.
2. Isolated BF16-safe Stage2 SFT runner implementation.
3. Data readiness and H20 path validation.
4. 30-step SFT gate before any 100-step run.
5. 100-step SFT gate before any 300-step run.
6. DPO and 500-step confirmation only if SFT gates pass.
