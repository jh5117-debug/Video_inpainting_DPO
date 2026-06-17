# Commands

PAI launcher used for the completed limit=100 gate:

```bash
cd /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO
CUDA_VISIBLE_DEVICES=0 RUN_VERSION=20260617_exp16_limit100 \
  bash exp16_prior_confidence_gated_dpo/scripts/launch_exp16_pai.sh \
  2>&1 | tee logs/pipelines/exp16_prior_confidence_limit100_20260617.log
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
bash exp16_prior_confidence_gated_dpo/scripts/launch_exp16_pai.sh
```

The launcher performs prior-cache check/generation, confidence audit, Stage1
preflight, and Stage1 500. It does not launch Stage2 or full training.
