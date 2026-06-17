# Exp18 Propagation Cache Quality Limit100

Status:

```text
PENDING_PAI_RUN
```

The cache quality report is not available in this HAL session because the PAI
training manifest and data mount are not visible here.

Expected command on PAI:

```bash
bash exp18_multiframe_propagation_gated_dpo/scripts/prepare_exp18_cache_limit100_pai.sh
```

Expected generated files:

```text
reports/exp18_propagation_cache_quality_limit100.md
reports/exp18_propagation_cache_quality_limit100.csv
/mnt/nas/hj/H20_Video_inpainting_DPO/data/cache/exp18_multiframe_propagation_cache_limit100/manifests/exp18_train_with_multiframe_prop_limit100.jsonl
```

Do not start Exp18 training until this report shows useful propagation coverage
and propagated-region quality.

