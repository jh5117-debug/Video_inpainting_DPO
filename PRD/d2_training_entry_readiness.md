# D2 Training Entry Readiness

Updated: 2026-05-30

## Ready Inputs

D2 generated-loser manifests are ready for training and evaluation reruns.
Do not regenerate the dataset in this pass.

| Split | Manifest | Rows |
| --- | --- | ---: |
| comp primary | `selected_primary_comp.repaired.jsonl` | 10000 |
| nocomp primary | `selected_primary_nocomp.repaired.jsonl` | 10000 |
| comp secondary | `selected_secondary_comp.repaired.jsonl` | 10000 |
| nocomp secondary | `selected_secondary_nocomp.repaired.jsonl` | 10000 |

Canonical PAI D2 root:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4
```

H20 may use a local path-rewritten `.h20.jsonl` manifest when the PAI absolute
paths are not mounted. This is a path adaptation only; it must not change the
training examples.

## Failed beta500 Entry

Old Exp5 beta500 is failed/collapsed and diagnostic only.

Configuration:

```text
manifest = selected_primary_comp.repaired.jsonl
train_mask_mode = full
mask_from_manifest = false
loss_region_mode = full
beta_dpo = 500
sft_reg_weight = 0
stage1_steps = 10000
stage2_steps = 10000
```

Failure mode:

- DPO objective saturated early (`acc=1`, `dpo=0`, `loss=0`).
- Stage2 10000 VBench and side-by-side videos show high-frequency noise and
  color explosion.
- This is an optimization/preference-data failure, not a task failure.

## beta10 Rerun Entries

| Experiment | Manifest | beta_dpo | Stage1 | Stage2 | Train mask | Mask from manifest | Loss region | Eval |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| `exp5_d2_comp_k4_beta10_s1s2_4000` | `selected_primary_comp.repaired.jsonl` | 10 | 4000 | 4000 | full | false | full | qual30 + full VBench |
| `exp6_d2_nocomp_k4_beta10_s1s2_4000` | `selected_primary_nocomp.repaired.jsonl` | 10 | 4000 | 4000 | full | false | full | qual30 + full VBench |

Validation during training stays disabled with `VAL_STEPS=999999`. Checkpoints
are saved every 1000 steps with `CKPT_LIMIT=2`.
