# Exp41 H20 MiniMax Parallel BF16

Status: `H20_MINIMAX_SFT_BLOCKED`

Exp41 is an H20-only parallel MiniMax adapter track. PAI remains read-only and
continues to own any active Exp40/PAI-side work. This branch does not modify
MiniMax official source files, `inference/metrics.py`, shared trainers, or
Exp1-Exp40 history.

## Scope

- Use H20 as a second server for MiniMax adapter debugging and gated training.
- Prefer existing code, flags, environment variables, launch wrappers, and run
  configs over source changes.
- If a BF16/SIGFPE fix requires source edits, stop and write a patch proposal
  instead of editing source.
- Do not write universal-adapter, final-SOTA, all-models-supported, or
  top-conference-novelty claims.

## Source Of Truth

- Branch: `research/exp41-h20-minimax-parallel-bf16-20260629`.
- Base: `origin/research/exp40-minimax-psnr-safe-rescue-20260628`.
- Start HEAD: `ecd82ef8bfefd1efba063d2a240631c1b7230b1d`.
- H20 worktree:
  `/home/nvme01/H20_Video_inpainting_DPO_exp41_minimax_h20_parallel`.
- HAL/local worktree:
  `/home/hj/H20_Video_inpainting_DPO_exp41_minimax_h20_parallel`.

## Current MiniMax State

- Exp30/35/36/37/38 show MiniMax is trainable and inference uses trained
  weights, but no heldout quality-positive adapter result exists.
- Exp38 R1 had a small raw PSNR signal but failed boundary/outside and visual
  safety.
- Exp40 built a VOR-Train LocalDPO v3 minimum pool:
  `train64/search24/shadow24`, zero split scene overlap, all selected rows
  `MEDIUM_HARD_ELIGIBLE`.
- Exp40 Step0 baseline is established on the LocalDPO v3 minimum pool.
- Exp39 H20 mirror/env smoke passed: required H20 MiniMax training/smoke paths
  are present, weights resolve, and H20 `wan` env imports MiniMax.

## H20 GPU Release

Initial H20 GPU audit found GPU4 occupied by a non-system compute task:

```text
PID 3365990, PGID 3365988
/home/nvme03/workspace/lingbot-world/.conda_envs/lingbot-world-v2/bin/python
-m cam_physgeo.dpo.failure_diagnostics sigma-sensitivity ...
```

The task was unrelated to Video_inpainting_DPO and was cleared with
`TERM -- -3365988`; no `KILL` was required. Final H20 GPU state has no compute
apps on GPU0-GPU7. `nvitop` still holds `/dev/nvidia*` handles but is not a
compute process.

Report:

- `reports/exp41_h20_gpu_release_audit.md`
- `reports/exp41_h20_gpu_release_audit.csv`


## H20 Data / Weight Audit

Status: `H20_MINIMAX_DATA_READY`

H20 mirror validation passed after filling missing evidence assets from PAI
read-only rsync. The final audit checked `2242` active refs across Exp30, Exp37,
Exp38, and Exp41/Exp40 manifests with `0` missing. Exp40 H20-safe LocalDPO v3
manifests are available for `64` train, `24` search, and `24` shadow rows.

Additional checks:

- Exp40 raw output decode: `112/112` readable.
- Exp40 direct VOR condition/winner/mask decode: `22/22` each readable.
- First-frame mask non-empty check: `0` empty masks.
- MiniMax scheduler/transformer/VAE files resolve via H20 `current` symlink.

This gate does not authorize training by itself. BF16/SIGFPE runtime preflight
and official protocol audit remain pending.

Reports:

- `reports/exp41_h20_minimax_data_audit.md`
- `reports/exp41_h20_minimax_manifest_validation.csv`
- `reports/exp41_h20_minimax_missing_assets.csv`
- `reports/exp41_h20_minimax_decode_audit.csv`


## BF16 / SIGFPE Runtime Preflight

Status: `H20_MINIMAX_BF16_SAFE_READY`

Exp41 ran P0-P7 on H20 using an Exp41-only helper and launcher. The helper kept
VAE encode/decode in fp32, ran MiniMax DiT bf16 where required, reduced losses
in fp32, disabled flash/memory-efficient SDPA where PyTorch exposes backend
toggles, and used xFormers/flash-attn disable env flags.

All cases passed: P0 torch bf16 backward, P1 VAE fp32 encode/decode, P2/P3 DiT
bf16 forward/backward, P4 fp32 one-batch train, P5 bf16-safe single-GPU train,
P6 bf16-safe DDP2 train, and P7 bf16-safe DDP8 train. Rank0 checkpoint
save/reload passed for P4-P7. No SIGFPE, OOM, CUDA error, NaN/Inf, or Xid was
observed. Final H20 GPU compute apps: none.

This is a runtime gate only. It does not claim MiniMax quality improvement and
does not replace the official MiniMax protocol audit.

Reports:

- `reports/exp41_h20_bf16_preflight.md`
- `reports/exp41_h20_bf16_preflight.csv`
- `reports/exp41_h20_bf16_preflight_summary.json`
- `reports/exp41_h20_bf16_preflight_rank_details.csv`

## Official MiniMax Protocol Audit

Status: `H20_MINIMAX_PROTOCOL_MATCHES_OFFICIAL`

Exp41 audited the H20 official MiniMax repo and current Exp40/H20 Step0
baseline runner. Official README Quick Start and `test_minimax_remover.py` use
`UniPCMultistepScheduler`, `float16`, `num_inference_steps=12`, and
`iterations=6`; current Exp40 Step0 defaults match those executable examples.
The README prose phrase "6 inference steps" is recorded as a documentation
ambiguity, so Exp41 ran a 6-step probe as diagnostic evidence only.

Smoke scope:

- `official_readme_test`: 4 train rows and 4 search rows at 12 steps /
  6 iterations.
- `feature_6step_probe`: the same rows at 6 steps / 6 iterations, diagnostic
  only.
- Raw output primary: true.
- Diagnostic comp used: false.
- Training launched: false.

Codex opened the pulled local contact sheets covering all `16` midframe review
sheets and all `16` temporal strips, and decoded all `16` side-by-side mp4s.
No mask polarity reversal, hidden comp, or winner/GT leakage into raw output was
observed. Several rows still show baseline quality issues, including
over-erasure, fog-like fill, terrain/shore hallucination, and masked-region dark
artifacts.

Decision: protocol identity passes. This is not a MiniMax quality-positive
result and not third-backbone evidence.

Reports:

- `reports/exp41_h20_minimax_official_protocol_audit.md`
- `reports/exp41_h20_minimax_official_protocol_audit.csv`
- `reports/exp41_h20_official_vs_current_visual_review.csv`
- `reports/exp41_h20_official_protocol_summary.json`
- `reports/exp41_h20_official_protocol_video_decode_audit.csv`

## Readback Decision

H20 has passed data/weight completeness, BF16/SIGFPE preflight, and official
protocol identity. It can proceed to the gated SFT-only bad-noise ladder after a
fresh milestone readback and GPU audit.

Next required statuses:

```text
H20_MINIMAX_SFT_100STEP_PASS / H20_MINIMAX_SFT_300STEP_PASS / PARETO_MIXED / NEGATIVE / BLOCKED
```

No DPO or 500-step lane is authorized until the SFT ladder produces a safe
checkpoint and passes its gate.

## Lane A SFT Bad-Noise Ladder

Status: `H20_MINIMAX_SFT_BLOCKED`

Lane A was audited after the protocol gate passed. H20 GPU0-GPU7 had no compute
apps at the fresh readback, but no SFT training was launched.

Blocker: the existing MiniMax training scripts do not expose a no-source-change
30/100/300-step SFT-only ladder.

- Exp35 winner-SFT positive control hard-caps `steps > 10`.
- Exp36 winner-SFT positive control with S0/S1 scopes hard-caps `steps > 10`.
- Exp35 rescue recipes are DPO-oriented and hard-cap `steps > 10`.
- Exp37 LocalDPO bad-noise recipes hard-cap DPO steps at 10 and SFT warmup at
  5.
- Exp38 and Exp40 scripts available in this branch are evaluators, not SFT
  trainers.

Under the no-source-code-modification rule, Exp41 cannot bypass those caps or
create a new training runner. A patch proposal was written instead.

Reports:

- `reports/exp41_h20_sft_badnoise_ladder.md`
- `reports/exp41_h20_sft_badnoise_ladder_metrics.csv`
- `reports/exp41_h20_sft_badnoise_ladder_visual_review.csv`
- `reports/exp41_h20_sft_badnoise_ladder_summary.json`
- `reports/exp41_h20_sft_ladder_patch_proposal.md`
