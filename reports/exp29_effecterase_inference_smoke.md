# Exp29 EffectErase Inference Smoke

Date: 2026-06-26

Status: `EFFECTERASE_INFERENCE_SMOKE_BLOCKED_NO_WEIGHTS`

The official EffectErase repository is locally available, but the required
official model assets were not found in the audited local or NAS paths.

Required by the official README / code:

- `EffectErase.ckpt`
- `Wan2.1-Fun-1.3B-InP` components
- Wan VAE / diffusion model / text encoder / CLIP assets

Because the official weights are missing, Exp29 did not run an EffectErase
inference smoke and did not substitute another checkpoint. EffectErase remains
an OR baseline / diagnostic candidate, not a true adapter candidate in this
run.

Data-risk reminder: EffectErase is VOR-trained, so VOR-train or VOR-Eval smoke
results cannot be used as primary on-policy scientific proof. They can only be
used as baseline / diagnostic evidence unless a non-confounded external OR
validation design is locked first.

