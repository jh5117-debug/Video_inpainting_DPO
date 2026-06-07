# Exp7c GT-Win Small-D2 Partial-Mask

Exp7c is the Exp7a control with the same small-D2 generated loser, mask,
training loss, steps, precision profile, and small-D2 validation, but with a
materialized GT/winner manifest generated from the original VideoDPO pair
source instead of reusing the opaque Exp7a `win_video_path` cache directly.

## Invariants

- Source loser/mask data: `exp07_fix_videodpo_smallmask15_20_prior_k4`
- Training dataset type: `generated_loser_manifest`
- Mask/task: `train_mask_mode=partial`, `mask_from_manifest=true`
- Loss: full-region DPO, beta 10, lose-gap 0.25, winner-abs 0.05,
  winner-gap ReLU weight 1.0
- Stage schedule: Stage1 2000 steps, Stage2 2000 steps
- Validation: VideoDPO small-D2 partial-mask eval, not DAVIS
- Validation checkpoints:
  - `exp7c-1`: Stage1-DPO spatial + SFT Stage2 motion hybrid
  - `exp7c-2`: Stage1-DPO + Stage2-DPO

## Code

- Manifest tool: `code/prepare_gtwin_manifest.py`
- One-shot H20 launcher: `code/launch_s1s2_h20.sh`

The launcher performs:

1. Prepare GT/winner manifest.
2. Run Stage1.
3. Build `exp7c-1` hybrid.
4. Run small-D2 eval for `exp7c-1`.
5. Run Stage2.
6. Run small-D2 eval for `exp7c-2`.

