# Exp23 GPU2/4/5/6 Pair Completion Report

Date: 2026-06-21

Status: `STAGE1_STAGE2_PAIR_COMPLETED`

## Run

```text
pair_id = phaseA_scale1_pair001_outer2_gpus2456
fresh Exp11 twin = fresh_exp11_outer_b075
candidate = candidate_scale1_outer2_b075
gpus = 2,4,5,6
nproc_per_node = 4
started_at = 2026-06-21 11:43:57 CST
finished_at = 2026-06-21 19:47:50 CST
```

## Outputs

Pair root:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp23_two_stage_pool_morphology_sweep/pairs/phaseA_scale1_pair001_outer2_gpus2456/
```

Candidate Stage2 final weights:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp23_two_stage_pool_morphology_sweep/pairs/phaseA_scale1_pair001_outer2_gpus2456/candidate_scale1_outer2_b075/stage2/last_weights
```

Fresh Exp11 Stage2 final weights:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp23_two_stage_pool_morphology_sweep/pairs/phaseA_scale1_pair001_outer2_gpus2456/fresh_exp11_outer_b075/stage2/last_weights
```

## Training Summary

| model | stage | last step | final total loss | final dpo loss | final grad norm | max grad norm | last loser-dominant | checkpoints | last_weights |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| fresh_exp11_outer_b075 | Stage1 | 2000 | 0.399996 | 0.392117 | 222.153356 | 728.881603 | 1.000000 | 1000,1500,2000 | yes |
| fresh_exp11_outer_b075 | Stage2 | 2000 | 0.404275 | 0.392957 | 7.555776 | 63.415727 | 1.000000 | 1000,1500,2000 | yes |
| candidate_scale1_outer2_b075 | Stage1 | 2000 | 0.432572 | 0.428459 | 74.767941 | 578.498537 | 1.000000 | 1000,1500,2000 | yes |
| candidate_scale1_outer2_b075 | Stage2 | 2000 | 0.498125 | 0.489977 | 7.075422 | 30.997587 | 1.000000 | 1000,1500,2000 | yes |

## Final GPU State

```text
GPU2 = 0 MiB, 0%
GPU4 = 244 MiB, 0%
GPU5 = 4 MiB, 0%
GPU6 = 292 MiB, 0%
```

No Exp23 `Phy` process remained after completion.

## Notes

- The GPU7 ghost allocation remained avoided throughout this run.
- The earlier Stage2 parser issue was fixed by adding `--aggregation` compatibility to the isolated Exp23 Stage2 trainer.
- Fresh Exp11 Stage2 was restarted after an external high-expert job caused OOM on GPU4/5/6. After terminating that conflicting job, fresh Stage2, candidate Stage1, and candidate Stage2 all completed.
- Both Stage1 runs showed large but finite gradient spikes. This should be reviewed before expanding the queue.
- Final loser-dominant ratio remained `1.0` in all four stages.
- DAVIS50 evaluation was not launched by the current runner. This report is a training-completion report, not an evaluation result.

## Next Gate

Run paired DAVIS50 evaluation for the fresh Exp11 twin and `candidate_scale1_outer2_b075` before making any quality claim or advancing the Exp23 sweep.
