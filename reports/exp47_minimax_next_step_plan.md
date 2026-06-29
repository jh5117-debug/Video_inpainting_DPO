# Exp47 MiniMax Next Step Plan

Recommended next action: `NEXT_H20_LOCAL_PSEUDOSUCCESS_TARGET_1_10STEP`

Do not run global pseudo-success SFT, 100-step SFT, DPO, or GT-only SFT from the current Exp46 setup.

## Localized Probe Requirements

- Input: `manifests/exp47_success_local_only.jsonl`.
- Construct target only in mask/boundary/affected local region; outside should anchor to Step0/condition or V_bg preservation, not pseudo-success full-frame tone.
- Remove global affected-map propagation from pseudo target drift.
- Use no more than 1/10 steps for first probe.
- Require Step30-style movement audit before any longer run.
- No MiniMax third-backbone claim unless heldout/shadow metrics and visual review pass later.

## Blocked Paths

- Global pseudo-success SFT: blocked by strict-clean count `0` and region contribution risk.
- GT-only SFT: blocked by Exp43 negative result and not part of Exp47.
- Same-source DPO: possible later, but not the immediate first repair because Step30 did not learn pseudo target and region objective must be localized first.
