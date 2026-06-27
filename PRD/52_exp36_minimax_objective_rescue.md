# Exp36 MiniMax Objective Rescue

Status: `EXP36_READBACK_COMPLETED`

Exp36 is an isolated MiniMax root-cause and objective-rescue track. It starts
from Exp35, where MiniMax was confirmed trainable and inference-sensitive but
where every bounded 10-step rescue recipe failed the heldout quality gate.

## Scope

- Diagnose whether the MiniMax failure is code/loading, trainable scope,
  learning-rate/update scale, bad-noise/timestep state, or objective design.
- Translate Linear-DPO, SDPO, and LocalDPO ideas into bounded, auditable
  MiniMax flow-DPO ablations.
- Run no GPU task until the readback milestone is committed.
- Run no 30-step confirmatory micro unless a new Exp36 10-step objective
  rescue gate passes.
- Do not run 500/1000/2000-step, RC-FPO, or universal-adapter claims.

## Protected Lanes

The following lanes are protected and read-only for Exp36:

- Exp31 VideoPainter 2000-step long run.
- Exp33 EffectErase VOR-Eval baseline.
- Left CLI / cli4 Exp25/Exp27/Exp28 worktrees, locks, heartbeats, and outputs.

Signals sent to protected lanes: `no`.
Protected files modified: `no`.

## 2026-06-27 Readback

- Branch: `research/exp36-minimax-objective-rescue-20260627`.
- Base: `origin/research/exp35-minimax-flow-dpo-rescue-20260627`.
- Start HEAD: `fb70266d53f5f9abd5e8d09ef9d2de324a10b7d6`.
- Worktree:
  `/home/hj/H20_Video_inpainting_DPO_exp36_minimax_objective_rescue`.
- Output root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp36_minimax_objective_rescue`.
- Log root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp36_minimax_objective_rescue`.

Previous MiniMax evidence read:

- Exp30 zero-gap and one-step strict reload passed.
- Exp30 frozen/EMA 10-step produced no heldout visual improvement.
- Exp35 no-change forensic audit found nonzero checkpoint/output movement but
  near-constant utility and sub-useful update scale.
- Exp35 inference-sensitivity perturbation confirmed inference consumes
  MiniMax transformer weights.
- Exp35 current trainable scope was full MiniMax transformer, not ignored LoRA.
- Exp35 winner-SFT positive-control proved trainability but harmed heldout
  quality.
- Exp35 hard-noise states were mined.
- Exp35 R1/R2/R3 10-step rescue recipes all failed quality gates.

Readback report:

- `reports/exp36_minimax_objective_rescue_readback.md`

Next eligible milestone:

- No-change forensic audit using previous Exp30/Exp35 checkpoints and logs,
  with no new training.

## 2026-06-27 No-Change Forensic Audit

Status: `MINIMAX_NOCHANGE_CAUSE_UTILITY_SCALE_TOO_WEAK`.

No new training or inference was launched. The audit reread Exp30/Exp35
checkpoint, output-diff, loss-scale, SFT-control, hard-noise, and rescue
recipe reports.

Conclusion:

- Checkpoint fallback is not supported: Step10 outputs are not byte-identical,
  keys match, and Exp35 sensitivity showed inference consumes weights.
- Trainable-scope failure is not supported by current evidence: Exp30/Exp35
  used the full MiniMax transformer scope, and updates moved outputs.
- Exp30 utility/margin was too weak, staying near `0.5` with tiny margins.
- Exp35 hard-noise rescue made stronger movement but moved in a harmful
  direction: all R1/R2/R3 heldout mask, boundary, and outside PSNR deltas were
  negative and visual better rows stayed `0/48`.

Reports:

- `reports/exp36_minimax_nochange_forensic_audit.md`
- `reports/exp36_minimax_nochange_param_delta.csv`
- `reports/exp36_minimax_nochange_output_diff.csv`
- `reports/exp36_minimax_nochange_loss_scale.csv`
- `reports/exp36_minimax_nochange_summary.json`

Next eligible milestone: Exp36 inference sensitivity test. No 30-step or long
training is unlocked.
