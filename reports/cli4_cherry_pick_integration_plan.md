# CLI4 Cherry-Pick Integration Plan

Generated: 2026-06-27

This file is an integration plan only. Do not auto-cherry-pick into the original Exp25, Exp26, Exp27, or Exp23 feature branches.

## Protected Branches

- Do not cherry-pick automatically into `research/exp26-videopainter-dpo-v2`.
- Do not touch the right-side Exp26 worktree or output root.
- Exp26 was read-only from this CLI session.

## Exp25 CLI Branch

Branch: `research/exp25-vor-gate16-cli4-20260625`

Worktree: `/home/hj/H20_Video_inpainting_DPO_exp25_cli4`

Candidate commits for future manual integration:

- `0e07cc8`
- `8aad8dd`
- `f2a4845`
- `918eedb`

Result summary:

- Gate16 used the reported DE-B stack.
- Gate16 passed: 16/16 technical valid, 7 medium-hard, 7 hard-plausible, 2 trivial-bad, 0 invalid.
- Status: `EXP25_DIFFUSERASER_GATE16_PASSED`.
- OR-DPO was not started by CLI4.

## Exp27 CLI Branch

Branch: `research/exp27-localdpo-objective-cli4-20260625`

Worktree: `/home/hj/H20_Video_inpainting_DPO_exp27_cli4`

Candidate commits for future manual integration:

- `d271747`
- `f1aa52d`
- `3c07f83`

Result summary:

- P8/P32 pair generation passed.
- LocalDPO objective failed after the allowed one fix/resume cycle.
- Status: `FAILED_FINAL`.
- RC-FPO remains `NOT_STARTED`.

## Exp28 CLI Branch

Branch: `research/exp28-fine-inner-boundary-sweep-20260625`

Worktree: `/home/hj/H20_Video_inpainting_DPO_exp28_inner_boundary`

Candidate commits for future manual integration:

- `2e594d2` - add Exp28 code/PRD/tests
- `f89ce9e` - eval root fix
- `9a6de60` - autoresearch eval root
- `4429daa` - guard optional DAVIS50 metrics
- `2922cb2` - record Pair B reduced metric result
- `83c70c4` - record Pair C failed-final reduced eval

Result summary:

- Pair A training completed; eval failed-final before optional metric guards could complete a full result.
- Pair B completed reduced DAVIS50 and visual assets; main Stage2-2000 metric result is mixed, not a scientific positive.
- Pair C completed training for fresh control and inner8 candidate through Stage2 checkpoint-2000.
- Pair C reduced Stage2-1000 eval is negative/mixed: PSNR delta -0.125200 dB, win rate 0.46, LPIPS +0.000075, Ewarp -0.024456.
- Pair C full eval stopped after NAS iowait recurred following one resume.

Current Exp28 decision:

```text
NO_INNER_RADIUS_POSITIVE
NO_SCIENTIFIC_POSITIVE
CURRENT_BEST_REMAINS_EXP11_LEGACY_OUTER_ONE_RING
```

## Suggested Manual Order

1. Review Exp25 commits first if OR candidate-pool planning is needed.
2. Review Exp27 commits separately; do not merge failed objective state into a production baseline without a follow-up fix.
3. Review Exp28 commits as a group, but keep `83c70c4` with `2922cb2` so the Pair B mixed result and Pair C failed-final evidence are not separated.

No automatic integration was performed by CLI4.
