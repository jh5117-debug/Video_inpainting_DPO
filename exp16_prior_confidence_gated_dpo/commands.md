# Commands

PAI launcher:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash exp16_prior_confidence_gated_dpo/scripts/launch_exp16_pai.sh
```

Build a small prior-cache readiness subset first:

```bash
python exp16_prior_confidence_gated_dpo/code/precompute_propainter_prior_cache.py \
  --input_manifest /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4/manifests/selected_primary_comp.repaired.pai_paths.jsonl \
  --output_root exp16_prior_confidence_gated_dpo/cache/exp16_propainter_prior_cache \
  --propainter_model_dir weights/propainter \
  --limit 100 \
  --resume
```

Run implementation preflight against a prior manifest:

```bash
python exp16_prior_confidence_gated_dpo/code/preflight_exp16.py \
  --manifest exp16_prior_confidence_gated_dpo/cache/exp16_propainter_prior_cache/manifests/exp16_train_with_prior.jsonl
```

