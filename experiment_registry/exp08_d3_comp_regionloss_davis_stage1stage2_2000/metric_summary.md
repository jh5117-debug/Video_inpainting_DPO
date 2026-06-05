# Metric Summary

Status: pending.

Validation must use `tools/run_inpainting_metric_eval.py`, which delegates metric computation to `inference/metrics.py`.

Expected outputs:

- `logs/target_eval/exp08_stage1_val_davis_<timestamp>/metrics/summary.csv`
- `logs/target_eval/exp08_stage2_val_davis_<timestamp>/metrics/summary.csv`

VBench is not used for Exp8.
