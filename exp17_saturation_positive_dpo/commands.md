# Commands

PAI overnight gate:

```bash
nohup bash exp17_saturation_positive_dpo/scripts/launch_exp17_overnight_pai.sh \
  > logs/pipelines/exp17_saturation_positive_overnight.log 2>&1 &
echo $! > logs/pipelines/exp17_saturation_positive_overnight.pid
```

The launcher runs:

1. Exp17a Stage1 1000 + DAVIS10 eval.
2. Exp17b Stage1 1000 + DAVIS10 eval.
3. Exp17c Stage1 1000 + DAVIS10 eval.
4. Metric summary and visual grids.

It does not launch Stage2.
