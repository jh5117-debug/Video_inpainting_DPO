# Commands

Start the isolated Exp12 adaptive outer-boundary run:

```bash
EXP12_OUTER_CUDA_VISIBLE_DEVICES=4,5,6,7 \
EXP12_OUTER_EVAL_GPU=4 \
bash scripts/launch_exp12_adaptive_outer_boundary_pai.sh
```

This launches only `exp12_batch_adaptive_outer_b075_s1s2_2000`.

Canonical eval protocol is fixed to:

`DAVIS50 raw6 hard-comp / D+G off / no PCM / frame-wise in-memory metric`
