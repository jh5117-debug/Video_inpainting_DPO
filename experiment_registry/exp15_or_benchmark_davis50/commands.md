# Commands

PAI entrypoint:

```bash
cd /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO
nohup bash exp15_or_benchmark_davis50/scripts/run_or_inference_davis50_all_methods_pai.sh \
  > logs/pipelines/exp15_or_benchmark_davis50.log 2>&1 &
echo $! > logs/pipelines/exp15_or_benchmark_davis50.pid
```

Metrics only:

```bash
bash exp15_or_benchmark_davis50/scripts/eval_or_davis50_metrics_pai.sh
```

Visual grids only:

```bash
bash exp15_or_benchmark_davis50/scripts/make_or_davis50_visual_grids_pai.sh
```

