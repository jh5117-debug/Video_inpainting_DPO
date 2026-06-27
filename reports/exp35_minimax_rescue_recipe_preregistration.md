# Exp35 MiniMax Rescue Recipe Preregistration

Status: `MINIMAX_RESCUE_RECIPES_PREREGISTERED`

This milestone preregisters the bounded 10-step rescue recipes only. No
training, inference, metrics, or video-quality promotion was run.

## Preconditions

- Root-cause status: `MINIMAX_NOCHANGE_CAUSE_UTILITY_SCALE_TOO_WEAK`.
- Inference sensitivity: `MINIMAX_INFERENCE_SENSITIVITY_PASS`.
- Trainable scope: `MINIMAX_TRAINABLE_SCOPE_CURRENT_OK`.
- Positive-control status:
  `MINIMAX_POSITIVE_CONTROL_PASS_HELDOUT_QUALITY_NEGATIVE`.
- Bad-noise status: `MINIMAX_BAD_NOISE_STATES_READY`.
- Train state manifest SHA256:
  `fbadd0d2565c4bb49245931742215c4d074c9834b369342398058b4ed9732047`.
- Heldout state manifest SHA256:
  `947f6c0f660229f1da92cb756ee7e03cda4b2215d1ae8f154999574b590ec1fb`.

The winner-SFT positive-control proved MiniMax trainability but harmed
heldout quality. Therefore the rescue grid is intentionally small and must
stop after 10 steps unless the preregistered gate passes.

## Fixed Data And Training State

- Train rows: locked Exp30 Gate64 V3 train32.
- Heldout rows: locked Exp30 Gate64 V3 heldout16.
- VOR-Eval use: false.
- Target: MiniMax flow velocity `epsilon - z0`.
- Trainable scope: `S0_current_full_transformer_from_Exp30`.
- Steps: `10`.
- Endpoint: Step10 only.
- Checkpoints: step0, step1, step5, step10.
- Strict reload: required for all saved checkpoints.
- LR: `1e-5`, selected as the least destructive winner-SFT positive-control
  recipe and capped below `1e-4`.
- Optimizer: AdamW, betas `(0.9, 0.95)`, eps `1e-8`, weight decay `1e-4`.
- Grad clip: `1.0`.
- Utility scale: `10`.
- Hard state policy: fixed `hard_state_A` per row for R1/R2/R3.
- No 30-step in this milestone.

## Active Recipes

### R1 LoVI-Linear-Frozen-HardNoise

- Reference: frozen Step0.
- Objective: Linear-DPO on region-weighted flow residuals.
- Hard state: `hard_state_A`.
- Utility scale: `10`.
- Winner anchor: none.
- Outside preservation loss: none beyond region weighting.

### R2 LoVI-Linear-EMA-HardNoise

- Reference: EMA reference.
- EMA decay: `0.995`, inherited from Exp30.
- Objective: Linear-DPO on region-weighted flow residuals.
- Hard state: `hard_state_A`.
- Utility scale: `10`.

### R3 WinnerAnchor-Linear-Hybrid

- Reference: frozen Step0.
- Objective: Linear-DPO on region-weighted flow residuals.
- Hard state: `hard_state_A`.
- Utility scale: `10`.
- Winner SFT anchor: `lambda_winner_sft=0.05`.
- Outside preservation: `lambda_outside=0.02`.

## Inactive Recipe

R4 `SDPO-Safe-Linear-Hybrid` is not active in this preregistration because
MiniMax flow-residual SDPO true-model parity has not been separately
validated. It must not be run as part of the next 10-step gate.

## Evaluation Gate

For each active recipe, evaluate heldout16 Step0 vs Step10 using the fixed
heldout state manifest and the existing MiniMax inference/metric protocol.

Pass requires all of:

- no NaN/Inf;
- strict reload pass;
- nontrivial output change;
- no collapse;
- heldout not worse globally;
- at least two local/effect metrics improve;
- outside not systematically worse;
- visual better >= `6/16`;
- worse/new-artifact <= `4/16`;
- Step10 not all tie/no-change.

Allowed statuses after the 10-step run:

- `MINIMAX_RESCUE_10STEP_RECIPE_PASS`
- `MINIMAX_RESCUE_10STEP_PARETO_MIXED`
- `MINIMAX_RESCUE_NO_OUTPUT_CHANGE`
- `MINIMAX_RESCUE_NEGATIVE`

Any 30-step confirmatory run remains forbidden unless the next milestone
returns `MINIMAX_RESCUE_10STEP_RECIPE_PASS`.
