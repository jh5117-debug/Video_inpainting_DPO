# Adapter Gate Candidate Recommendation

Status: prepared recommendation only. No training launched.

## Current Scope

Only two baselines remain in scope for the next adapter discussion:

1. VideoPainter
2. MiniMax-Remover

All other previously listed baselines are out of scope for this phase.

## Recommendation

### VideoPainter

Decision: **possible future adapter gate, but needs modification before training**.

Exp14 status:

```text
direct_diff_dpo_blocked_pending_isolated_trainer
```

Why it is the only realistic adapter-gate candidate now:

- public training scripts are available;
- local clone has train entrypoints;
- architecture is diffusion/DiT based;
- it is video inpainting/editing oriented.

Why not launch yet:

- data format conversion is required;
- DPO policy/reference integration is non-trivial;
- compute cost is high;
- the dedicated adapter trainer is not implemented yet;
- user requested skipping Smoke1 and Smoke20;
- the minimum preflight still cannot run because the isolated trainer is missing.

### MiniMax-Remover

Decision: **frozen baseline only for now**.

Why:

- public/local code available to us is inference-oriented;
- no train script or training-data path was validated;
- do not call it a trainable adapter until official training code is found.

## Launch Decision

Do not launch anything now.

If the user later approves an adapter gate, first implement an isolated
VideoPainter DPO adapter trainer under `exp14_adapter_videopainter/code/`, then
run the minimum preflight on PAI. Only after that passes should Gate2000 be
launched.
