# Exp20 Commands Used

Representative commands:

```bash
python -m py_compile exp20_autoresearch_scale_adaptive_region_dpo/code/*.py
python -m unittest discover -s exp20_autoresearch_scale_adaptive_region_dpo/tests -p 'test_*.py'
bash -n exp20_autoresearch_scale_adaptive_region_dpo/scripts/*.sh
```

PAI trial runner was used for fixed/adaptive/region/equal-step trials. Full VFID/TC/Ewarp backfills used:

```bash
python exp20_autoresearch_scale_adaptive_region_dpo/code/backfill_existing_eval_metrics.py \
  --compute-lpips --compute-vfid --compute-tc --compute-ewarp
```

Multiseed shadow confirmation used the isolated Exp20 scripts:

```bash
bash exp20_autoresearch_scale_adaptive_region_dpo/scripts/run_shadow_dev_baselines_pai.sh
bash exp20_autoresearch_scale_adaptive_region_dpo/multiseed_equal_step_confirmation/run_queue_worker.sh <gpu_id>
bash exp20_autoresearch_scale_adaptive_region_dpo/scripts/evaluate_multiseed_shadow_pai.sh
bash exp20_autoresearch_scale_adaptive_region_dpo/scripts/backfill_multiseed_candidate_metrics_pai.sh
bash exp20_autoresearch_scale_adaptive_region_dpo/scripts/run_multiseed_confirmation_postqueue_pai.sh
python exp20_autoresearch_scale_adaptive_region_dpo/code/analyze_bf07_p4_confirmation.py
python exp20_autoresearch_scale_adaptive_region_dpo/code/make_bf07_p4_visual_pack.py
```

No 500-step, 1000-step, 2000-step, Stage2, DAVIS50, or YouTubeVOS100 command was launched after the multiseed shadow gate failed.
