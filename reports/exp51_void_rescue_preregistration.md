# Exp51 VOID Loser-Dominant Rescue Preregistration

Status: `VOID_RESCUE_RECIPES_PREREGISTERED`

## Rationale

Exp50 10-step reduced DPO loss mainly by pushing the loser branch worse, while heldout video did not improve. Exp51 will therefore test winner-preserving, loser-clipped, local-only, and v-prediction-safe variants before any further 10-step run.

## Recipes

- R0 current baseline replay: existing Exp50 vanilla LoVI-DPO, not re-run by default.
- R1 WinnerPreserve-LocalDPO: local DPO on object+affected; winner anchor 0.05; outside 0.10; boundary 0.05; loser_grad_scale 0.0; proj_out only.
- R2 WinnerPreserve-LoserClip: object+affected+boundary; winner anchor 0.05; outside 0.10; loser_gap_clip_tau 0.0005; loser_grad_scale 0.1; proj_out only.
- R3 SDPO-Safe: preserve winner gradient and scale loser branch by safe lambda 0.1 if gradient geometry is available; proj_out only.
- R4 LinearDPO-vPrediction: replace sigmoid utility with linear v-prediction diagnostic; frozen reference first; proj_out only.
- R5 LocalDPO + LastBlock LoRA: only if R1-R4 fail one-step and memory is safe; rank 4; requires new zero-gap/one-step.

## Gates

One-step must be video-safe before any 10-step: full/outside PSNR >= Step0 - 0.02, visual better/tie >= 3/4, worse <= 1/4, no collapse, no tone drift, no boundary destruction.

10-step must keep full PSNR safe, avoid LPIPS/Ewarp regression, improve at least two local/effect metrics, and pass heldout visual review. No 30/50/100-step training is allowed.

## Safety

No VOR-Eval, hard comp, long training, official source edits, shared trainer edits, or `inference/metrics.py` edits are allowed.
