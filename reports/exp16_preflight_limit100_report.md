# Exp16 Preflight Limit100 Report

- generated_at: Wed Jun 17 14:34:23 CST 2026
- status: passed
- prior_manifest: `/mnt/nas/hj/H20_Video_inpainting_DPO/data/cache/exp16_propainter_prior_cache_limit100/manifests/exp16_train_with_prior_limit100.jsonl`
- output_dir: `exp16_prior_confidence_gated_dpo/runs/preflight_limit100`
- dpo_diag: `exp16_prior_confidence_gated_dpo/dpo_diag/preflight_dpo_diagnostics.csv`

The preflight uses the isolated Exp16 trainer, real ProPainter prior frames,
VAE-encoded latent targets, and reconstructed predicted x0 latent. It is not a
frozen-reference epsilon proxy.
