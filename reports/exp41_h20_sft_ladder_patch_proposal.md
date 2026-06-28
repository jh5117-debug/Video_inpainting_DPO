# Exp41 H20 MiniMax SFT Ladder Patch Proposal

Status: `PATCH_PROPOSAL_ONLY_NOT_APPLIED`

## Why A Patch Is Required

Lane A requires 30-step, 100-step, and conditional 300-step SFT-only training.
All existing MiniMax SFT/DPO runners in this branch are micro-gate scripts:

- Exp35 winner-SFT: capped at 10 steps.
- Exp36 winner-SFT with S0/S1 scopes: capped at 10 steps.
- Exp35 rescue DPO: capped at 10 steps and not SFT-only.
- Exp37 LocalDPO bad-noise: capped at 10 DPO steps and 5 warmup steps.

Running Lane A without a patch would require bypassing those caps or creating a
new training entrypoint. That is a source/training-runner change, so Exp41 stops
here under the current no-source-code-modification rule.

## Proposed Minimal Patch

Add an Exp41-isolated runner:

```text
exp41_h20_minimax_parallel_bf16/train_sft_badnoise_ladder.py
```

The runner should reuse existing audited utilities instead of changing shared
trainer code:

- `BatchCache`, `run_pipeline`, and metric helpers from Exp30/40;
- `load_transformer_for_scope` and S0/S1 scope handling from Exp36;
- BF16-safe dtype policy from Exp41 preflight;
- official protocol settings from Exp41 protocol audit.

Required config fields:

- `recipe`: `SFT-A`, `SFT-B`, or `SFT-C`;
- `steps`: `30`, `100`, or `300`;
- `lr`: `3e-5`, `1e-4`, or `3e-4`;
- `scope`: existing supported scope only;
- `mask_weight`, `boundary_weight`, `affected_weight`,
  `outside_preserve_weight`, `far_outside_weight`;
- `train_manifest`, `search_manifest`, `shadow_manifest`;
- `checkpoint_interval`;
- `bf16_safe`: true.

Required outputs:

- checkpoint-0, checkpoint-30, checkpoint-100, checkpoint-300 when applicable;
- diagnostics CSV with loss, grad norm, NaN/Inf, and checkpoint identity;
- train/search/shadow metrics CSV;
- visual review CSV;
- 16-frame temporal strips and side-by-side mp4s;
- summary JSON with gate decision.

## Guardrails

- Do not modify MiniMax official source files.
- Do not modify `inference/metrics.py`.
- Do not modify shared trainers.
- Do not alter Exp1-Exp40 history.
- Keep raw output primary and comp diagnostic only.
- Do not claim MiniMax positive without metrics plus visual review.
