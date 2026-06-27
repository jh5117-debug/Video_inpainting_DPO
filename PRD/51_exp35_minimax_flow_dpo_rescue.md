# Exp35 MiniMax Flow-DPO Rescue

Status: `EXP35_READBACK_COMPLETED`

Exp35 is an isolated rescue/debug track for the MiniMax flow-DPO adapter path.
It starts from Exp30 Gate64 V3 data readiness and the failed MiniMax 10-step
quality gate. The goal is not to add steps blindly, but to determine whether
the no-change behavior comes from checkpoint loading, inference sensitivity,
trainable scope, objective scale, learning rate, or missing bad-noise/hard
timestep selection.

## Ground Rules

- Do not continue VideoPainter 100-step or Exp31 2000-step.
- Do not touch Exp33 EffectErase VOR-Eval baseline.
- Do not touch left CLI / cli4 worktrees, locks, heartbeats, or processes.
- Do not modify `inference/metrics.py`, shared trainers, or Exp1-Exp29
  history.
- Do not overwrite Exp30 outputs, checkpoints, or reports.
- Do not run 30-step before an explicitly positive 10-step rescue recipe.
- Do not run 500/1000/2000-step, RC-FPO, or universal-adapter claims.

## 2026-06-27 Readback

- Branch: `research/exp35-minimax-flow-dpo-rescue-20260627`.
- Base branch: `origin/research/exp30-vor-or-multimodel-minimax-adapter-20260627`.
- Start HEAD: `f69688fe4ff96c4d4f0dcd308eef69822fc1035b`.
- Exp30 candidate-generation status: `VOR_OR_GATE64_MULTIMODEL_POOL_READY`.
- Exp30 MiniMax status: `MINIMAX_ADAPTER_RECIPE_NOT_READY`.
- Gate64 train32 SHA256:
  `1eda205d2dc48714269f30eb390d959549387a778e6438267e6aba087ba14196`.
- Gate64 heldout16 SHA256:
  `84c231ded930d740bf299b27c2a6b1e95d7decdb3665051371c5df90ae9f2ade`.

Exp30 failed as a quality-positive MiniMax adapter recipe, not as a data or
basic plumbing gate. The data pool exists, zero-gap passed, one-step strict
reload passed, no NaN/Inf occurred, and Step10 inference did load the saved
checkpoint. However, output movement was too small to be useful:

- Frozen mean heldout deltas: mask PSNR `-0.001068`, boundary PSNR
  `-0.002821`, outside PSNR `-0.006340`.
- EMA mean heldout deltas: mask PSNR `-0.001851`, boundary PSNR `-0.003092`,
  outside PSNR `-0.006033`.
- Codex visual review in Exp30: Step10 better `0/32`, tie/no visible
  improvement `32/32`, new visible artifact `0/32`.

Initial readback of the Exp30 script shows that the previous gate trained the
full MiniMax transformer, not a LoRA-only adapter scope. The recipe used
AdamW, LR `5e-7`, beta `1.0`, linear utility near `0.5`, deterministic
single-state timestep/noise per step, and no bounded bad-noise / hard-timestep
miner. The Step10 delta probe was only about `2.7e-7`, consistent with an
update-scale/objective-state problem.

Protected-lane readback:

- Exp31 was observed on GPU1, PID `1215136`, and is reserved.
- cli4 locks reserve GPU1/GPU2/GPU3/GPU4.
- Exp35 must only use eligible non-reserved GPUs and must not send signals to
  protected lanes.

Report:

- `reports/exp35_minimax_rescue_readback.md`

No GPU training, inference, 30-step, RC-FPO, or protected-lane action was
launched by this readback milestone.

