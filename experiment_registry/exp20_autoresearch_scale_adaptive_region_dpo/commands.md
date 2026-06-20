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
