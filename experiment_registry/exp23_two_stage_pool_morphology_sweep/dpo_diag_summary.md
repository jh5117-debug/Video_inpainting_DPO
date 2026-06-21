# Exp23 DPO Diagnostics Summary

Status: `STAGE1_STAGE2_PAIR_COMPLETED`

Pair: `phaseA_scale1_pair001_outer2_gpus2456`

GPU mapping: `2,4,5,6`

## Completed Training Diagnostics

| model | stage | last step | final total loss | final dpo loss | final grad norm | max grad norm | last implicit acc | last loser-dominant |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fresh_exp11_outer_b075 | Stage1 | 2000 | 0.399996 | 0.392117 | 222.153356 | 728.881603 | 0.750000 | 1.000000 |
| fresh_exp11_outer_b075 | Stage2 | 2000 | 0.404275 | 0.392957 | 7.555776 | 63.415727 | 1.000000 | 1.000000 |
| candidate_scale1_outer2_b075 | Stage1 | 2000 | 0.432572 | 0.428459 | 74.767941 | 578.498537 | 1.000000 | 1.000000 |
| candidate_scale1_outer2_b075 | Stage2 | 2000 | 0.498125 | 0.489977 | 7.075422 | 30.997587 | 1.000000 | 1.000000 |

## Interpretation

- All four stages reached step 2000 and wrote `last_weights`.
- Stage1 had large finite gradient spikes for both fresh Exp11 and candidate.
- Stage2 gradients were lower but still showed finite spikes.
- Final loser-dominant ratio stayed at `1.0`, so this pair is not clean by DPO diagnostics.
- These are training diagnostics only. DAVIS50 evaluation has not run yet.
