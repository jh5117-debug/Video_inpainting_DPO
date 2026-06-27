# Exp35 Status

Current status: `MINIMAX_POSITIVE_CONTROL_PASS_HELDOUT_QUALITY_NEGATIVE`

## 2026-06-27 Readback

- Branch: `research/exp35-minimax-flow-dpo-rescue-20260627`.
- Base HEAD: `f69688fe4ff96c4d4f0dcd308eef69822fc1035b`.
- Exp30 Gate64 pool: `VOR_OR_GATE64_MULTIMODEL_POOL_READY`.
- Exp30 MiniMax gate: `MINIMAX_ADAPTER_RECIPE_NOT_READY`.
- Failure class at readback: recipe/update-state suspected, not data
  availability or basic plumbing.
- Protected lanes checked read-only.
- No GPU inference, training, 30-step, RC-FPO, or protected-lane action
  launched.

Report:

- `reports/exp35_minimax_rescue_readback.md`

## 2026-06-27 No-Change Forensic Audit

- Status: `MINIMAX_NOCHANGE_CAUSE_UTILITY_SCALE_TOO_WEAK`.
- Training launched: false.
- Checkpoint keys: 461 common keys and 0 missing/unexpected keys for both
  frozen and EMA recipes.
- Frozen parameter delta/param norm ratio: `5.6404525516172905e-06`.
- EMA parameter delta/param norm ratio: `5.630459939756668e-06`.
- Step0/Step10 byte-identical rows: 0/32.
- Mean full/mask/affected/outside abs pixel diff:
  `0.13143352206508793`, `0.18672874342540607`,
  `0.1731182035360047`, `0.10850902535158265`.
- Frozen linear utility mean/min/max:
  `0.4999982982873917` / `0.49997058510780334` /
  `0.5000085830688477`.
- EMA linear utility mean/min/max:
  `0.5000003516674042` / `0.49999284744262695` /
  `0.5000050663948059`.
- Decision: run inference-sensitivity positive-control before recipe redesign.

## 2026-06-27 Inference Sensitivity Positive-Control

- Status: `MINIMAX_INFERENCE_SENSITIVITY_PASS`.
- Training launched: false.
- GPU used: PAI GPU6.
- Identity control max full MAE: `0.0`.
- Perturbed mean full/mask MAE: `0.08821829589193357` /
  `0.15630244233590715`.
- Codex visual review: `4/4` comparison strips opened; all showed subtle
  nonzero response with no collapse or new artifact.

This confirms that MiniMax inference consumes the transformer checkpoint
weights. It is not a quality-positive adapter result.

## 2026-06-27 Trainable-Scope Audit

- Status: `MINIMAX_TRAINABLE_SCOPE_CURRENT_OK`.
- Training launched: false.
- GPU used: none.
- Checkpoint tensors: `461`.
- Total parameters represented: `1127055424`.
- LoRA/adapter tensors: `0`.
- Exp30 scope: `all_transformer_parameters`.

The current scope is in the inference path and is not too small. No expanded
LoRA scope was prepared in this milestone.

## 2026-06-27 Winner-SFT Positive-Control

- Status: `MINIMAX_POSITIVE_CONTROL_PASS_HELDOUT_QUALITY_NEGATIVE`.
- Training type: winner reconstruction SFT positive-control, not DPO.
- GPU used: PAI GPU6.
- Steps: `10` per recipe.
- Recipes: AdamW LR `1e-5`, `3e-5`, `1e-4`.
- Best training-loss recipe: `1e-5`, loss `0.7092440128 -> 0.0127931200`.
- All recipes produced nonzero parameter/output changes and no NaN/Inf.
- Heldout quality was negative: mean mask PSNR deltas were `-0.244838`,
  `-0.889703`, and `-4.261956` for LR `1e-5`, `3e-5`, and `1e-4`.
- Codex opened `12/12` heldout temporal strips. Visual review found no clear
  quality-positive rows; LR `1e-4` introduced strong new artifacts on all 4
  heldout rows.

Conclusion: MiniMax is trainable and not frozen, but naive winner-SFT
overfits/harms heldout outputs. This does not unlock 30-step training or a
third-backbone positive claim.

## 2026-06-27 Bad-Noise / Hard-Timestep Miner

- Status: `MINIMAX_BAD_NOISE_STATES_READY`.
- Training launched: false.
- GPU used: PAI GPU0.
- Train rows: `32`.
- Heldout rows: `16`.
- Candidate states per row: `16`.
- Timesteps: `0.15`, `0.35`, `0.55`, `0.75`.
- CSV rows: `768`.
- Train state manifest SHA256:
  `fbadd0d2565c4bb49245931742215c4d074c9834b369342398058b4ed9732047`.
- Heldout state manifest SHA256:
  `947f6c0f660229f1da92cb756ee7e03cda4b2215d1ae8f154999574b590ec1fb`.

The miner selected `hard_state_A`, `hard_state_B`, and `hard_state_C` for each
row using frozen Step0 residuals. This prepares bounded 10-step rescue recipe
testing only. It is not a quality-positive adapter result and does not unlock
30-step training.

## 2026-06-27 Rescue Recipe Preregistration

- Status: `MINIMAX_RESCUE_RECIPES_PREREGISTERED`.
- Training launched: false.
- Inference launched: false.
- Active recipes: `R1`, `R2`, `R3`.
- Inactive recipe: `R4` SDPO-safe hybrid, blocked until MiniMax SDPO
  true-model parity exists.
- Steps: `10`.
- LR: `1e-5`.
- Utility scale: `10`.
- Hard-state policy: fixed `hard_state_A`.
- 30-step unlocked: false.

The next eligible milestone is the preregistered 10-step recipe run. It must
produce real heldout videos, metrics, strict reload evidence, and per-video
review before any recipe pass or quality language is allowed.
