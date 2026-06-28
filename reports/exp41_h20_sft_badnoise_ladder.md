# Exp41 H20 MiniMax SFT Bad-Noise Ladder

Status: `H20_MINIMAX_SFT_BLOCKED`

## Scope

This milestone attempted to start Lane A, the bad-noise robust SFT-only ladder,
after Exp41 passed data readiness, BF16/SIGFPE preflight, and official MiniMax
protocol identity.

No SFT/DPO training was launched in this milestone.

## Fresh Readback

- Local branch: `research/exp41-h20-minimax-parallel-bf16-20260629`.
- Local start HEAD: `d6666e2b501f93a194f756482cc2f3f51589f512`.
- H20 worktree HEAD: `9471b6f69d4a4910f57e05ae2da7e705a71eb255` at GPU readback,
  then fast-forwarded to `d6666e2b501f93a194f756482cc2f3f51589f512`.
- H20 GPU0-GPU7 compute apps at readback: none.
- PAI policy: read-only only; no PAI GPU, signal, file mutation, or runtime
  mutation.

## Existing Runner Audit

The requested Lane A requires 30-step SFT gate, 100-step continuation for
passing configs, and 300-step only if 100-step passes. It also requires
SFT-A/B/C style loss variants and train/search/shadow evaluation.

Existing code does not provide a legal no-source-change entrypoint for that:

| path | role | audited capability | blocker |
| --- | --- | --- | --- |
| `exp35_minimax_flow_dpo_rescue/scripts/run_minimax_winner_sft_positive_control.py` | winner-SFT micro positive control | SFT-only winner reconstruction | hard raises when `steps > 10`; fixed micro-gate weights, not SFT-A/B/C |
| `exp36_minimax_objective_rescue/scripts/run_minimax_winner_sft_positive_control.py` | winner-SFT micro positive control with S0/S1 scopes | supports S0 full transformer and S1 LoRA-like scope | hard raises when `steps > 10`; fixed micro-gate weights, not 30/100/300 ladder |
| `exp35_minimax_flow_dpo_rescue/scripts/run_minimax_rescue_10step_recipes.py` | Linear-DPO rescue recipes | DPO recipes only | hard raises when `steps > 10`; not SFT-only |
| `exp37_minimax_localdpo_badnoise_rescue/scripts/run_exp37_minimax_localdpo_badnoise_10step.py` | LocalDPO bad-noise recipes | 10-step LocalDPO and at most 5-step SFT warmup | hard raises when DPO `steps > 10` or warmup `> 5`; not Lane A |
| `exp38_minimax_full_adapter_breakthrough/scripts/evaluate_minimax_existing_checkpoint.py` | evaluator | evaluates existing checkpoint on train/heldout | inference only; no training |
| `exp40_minimax_psnr_safe_rescue/scripts/run_step0_baseline.py` | baseline evaluator | Step0 raw baseline inference | inference only; no training |

## Decision

Lane A is blocked under the no-source-code-modification rule.

Starting a 30/100/300-step SFT ladder would require a new or modified training
runner. The prompt allows Exp41 helpers only for launcher/audit/manifest/report
work, not new training internals. Therefore Exp41 did not start SFT training and
did not attempt to bypass the existing 10-step caps.

## Required Patch Proposal

See `reports/exp41_h20_sft_ladder_patch_proposal.md`.

Minimum required change:

- add an Exp41-isolated SFT ladder runner or explicitly authorize modifying an
  existing Exp35/36 runner;
- preserve the already-audited BF16-safe policy;
- support SFT-A/B/C weights from config;
- support `steps=30/100/300` with checkpoints;
- evaluate train/search/shadow with raw output primary;
- emit full per-video metrics, visual review CSV, and temporal review assets.

No `UNIVERSAL_ADAPTER`, `FINAL_SOTA`, `TOP_CONFERENCE_NOVELTY_CONFIRMED`, or
MiniMax third-backbone claim is made.
