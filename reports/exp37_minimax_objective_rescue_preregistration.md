# Exp37 MiniMax Objective Rescue Preregistration

Status: `MINIMAX_OBJECTIVE_RESCUE_RECIPES_PREREGISTERED`

This milestone locks the LocalDPO-badnoise 10-step rescue recipes before any
training. It uses the Exp37 LocalDPO-style train32/heldout16 pool and the
Exp37 bad-noise state manifest.

No training, inference, checkpoint update, or model evaluation was launched by
this preregistration milestone.

## Locked Inputs

- Train manifest:
  `exp37_minimax_localdpo_badnoise_rescue/manifests/localdpo_or_train32.jsonl`
- Heldout manifest:
  `exp37_minimax_localdpo_badnoise_rescue/manifests/localdpo_or_heldout16.jsonl`
- Bad-noise states:
  `exp37_minimax_localdpo_badnoise_rescue/manifests/exp37_badnoise_states.jsonl`
- Bad-noise SHA256:
  `492210b2cd725faa348adcbafaf37bf82cc6790b4eb0607b9f758047d1c795d4`
- Trainable scope: Exp36 S1 LoRA attention/projection scope.
- Base LR: `1e-5`, from the best Exp36 winner-SFT control.
- Utility scale: `18.0`, fixed from the diagnostic ratio
  `10 / 0.570900 ~= 17.52`, rounded to a single preregistered value.

## Recipes

### R1 LocalDPO-Linear-HardNoise

- Objective: Linear-DPO frozen reference.
- Data: Exp37 LocalDPO-style corruption pairs.
- Region: local mask/affected/boundary region only.
- Noise state: `hard_state_A`.
- Utility scale: `18.0`.
- LR: `1e-5`.
- Winner anchor: `0.05`.
- Outside preservation: `0.02`.
- Steps: `10`.

### R2 LocalDPO-Linear-SDPO

- Objective: R1 plus SDPO safe-lambda on loser branch.
- Enable condition: only if the pre-run geometry check finds finite gradients,
  safe lambda in `[0, 1]`, and no winner-risk violation for the selected
  mini-batch.
- Noise state: `hard_state_A`.
- Utility scale: `18.0`.
- LR: `1e-5`.
- Winner anchor: `0.05`.
- Outside preservation: `0.02`.
- Steps: `10`.

If the geometry check fails, R2 is skipped and must be reported as
`SDPO_GEOMETRY_UNSTABLE_SKIP`, not silently replaced.

### R3 LocalDPO-SFTWarmup-Linear

- Warmup: `5` winner-SFT local-anchor steps.
- Main phase: `10` Linear-DPO steps.
- Data: same train32/heldout16 manifests.
- Noise state: `hard_state_A`.
- Utility scale: `18.0`.
- LR: `1e-5`.
- Winner anchor: `0.05`.
- Outside preservation: `0.02`.

## Fixed 10-Step Gate

All recipes must be evaluated on the locked heldout16 endpoint:

- PSNR
- SSIM
- LPIPS
- Ewarp
- mask PSNR
- boundary PSNR
- affected PSNR
- outside preservation
- temporal flicker
- object residual
- effect residual

Pass criteria:

- heldout visual better rows `>= 6/16`
- heldout visual worse rows `<= 4/16`
- at least two local/effect metrics improve
- LPIPS not worse by more than `0.001`
- Ewarp not worse by more than `0.05`
- outside preservation not systematically worse
- Step10 not visually identical to Step0

Only `MINIMAX_10STEP_LOCALDPO_BADNOISE_POSITIVE` unlocks the conditional
30-step milestone. Pareto-mixed, no-change, blocked, or negative outcomes do
not unlock 30-step.

## Still Forbidden

- Repeating Exp36 failed recipes without the LocalDPO/bad-noise changes.
- Blind 30-step.
- 2000-step or other long training.
- RC-FPO.
- Universal-adapter or all-models-supported claims.
