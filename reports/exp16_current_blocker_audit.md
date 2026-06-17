# Exp16 Current Blocker Audit

Date: 2026-06-17

## Scope

Exp16 is the only active branch for this pass. OR, adapter work, adaptive
normalization, and Exp11/Exp12 tuning are paused.

## Current Status

```text
current_status = stage1_500_limit100_completed
```

Exp16 has an isolated folder:

```text
exp16_prior_confidence_gated_dpo/
```

Current implemented pieces:

- `precompute_propainter_prior_cache.py` can build a real ProPainter prior
  cache and write an Exp16 manifest with prior paths.
- `exp16_dataset.py` requires a real ProPainter prior path and refuses rows
  without it.
- `exp16_loss.py` implements GT-error prior confidence, outer-boundary masks,
  predicted-x0 reconstruction, and latent-space `L_prior`, `L_gen`,
  `L_boundary_extra`.
- `train_exp16_stage1.py` is wired to compute:
  `L_total = L_base + lambda_prior * L_prior + lambda_gen * L_gen +
  lambda_boundary_extra * L_boundary_extra`.
- `train_exp16_stage1.py --preflight_only` performs one real forward/backward
  and writes diagnostics before optimizer step.
- `launch_exp16_pai.sh` now runs limit=100 prior-cache generation, confidence
  audit, real Stage1 preflight, and then Stage1 500 only if preflight passes.

Resolved in this pass:

- Existing generated-loser manifests were audited and did not expose verified
  ProPainter prior fields.
- A real `limit=100` ProPainter prior cache was generated on PAI.
- Prior-confidence statistics were computed from real ProPainter prior frames
  against GT.
- Stage1 preflight passed with real ProPainter prior frames, VAE-encoded latent
  targets, reconstructed predicted latent x0, and one diagnostic row.
- Stage1 500 on the `limit=100` cache completed and saved `checkpoint-250`,
  `checkpoint-500`, `last_weights`, and `dpo_diagnostics.csv`.

Remaining blockers / guardrails:

- Stage2 is not wired for Exp16 prior-confidence loss and must remain disabled.
- Full 2000+2000 training is not authorized by this audit.
- DAVIS / YouTubeVOS evaluation has not been run for Exp16.
- This is an engineering small gate, not a final method result.

## Answers

1. Exp16 folder exists: yes.
2. `precompute_propainter_prior_cache.py` exists: yes.
3. `train_exp16_stage1.py` exists and is wired for real prior-x0 loss: yes.
4. `train_exp16_stage2.py` exists but is not wired/authorized: yes, blocked.
5. Current data can read ProPainter prior from the generated limit=100 Exp16
   manifest.
6. Stage1 can reconstruct predicted latent x0 through scheduler
   `prediction_type` and `alphas_cumprod`.
7. dpo_diag has Exp16 fields for prior/gen/boundary/confidence diagnostics.
8. Stage1 500 completed; Stage2 remains blocked.

## PAI Artifacts

Prior cache:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/data/cache/exp16_propainter_prior_cache_limit100/
```

Prior manifest:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/data/cache/exp16_propainter_prior_cache_limit100/manifests/exp16_train_with_prior_limit100.jsonl
```

Stage1 500 run:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260617_exp16_limit100_exp16_prior_confidence_s1_500_limit100_pai
```

Local diagnostic copies:

```text
exp16_prior_confidence_gated_dpo/dpo_diag/preflight_dpo_diagnostics.csv
exp16_prior_confidence_gated_dpo/dpo_diag/stage1_500_dpo_diagnostics.csv
```
