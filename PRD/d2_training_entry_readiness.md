# D2 Training Entry Readiness

## Data Asset

D2 is complete on PAI:

```text
root = /mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4
candidate_rows = 40000 / 40000
selected_primary_comp = 10000
selected_primary_nocomp = 10000
failed_shards = 0
generation_source = diffueraser_only after repair
```

Before training, run:

```bash
python tools/d2_post_generation_audit_and_repair.py \
  --output_root /mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4
```

Use repaired manifests only:

```text
Experiment 5: manifests/selected_primary_comp.repaired.jsonl
Experiment 6: manifests/selected_primary_nocomp.repaired.jsonl
Experiment 7: manifests/selected_primary_comp.repaired.jsonl + mask_path
Experiment 8: manifests/selected_primary_comp.repaired.jsonl + mask_path
```

## Current Training-Code Support

Current code supports:

- `--dpo_dataset_type diffueraser_inpainting`
- `--dpo_dataset_type videodpo_fullmask`
- full-video DPO loss through `compute_dpo_loss`
- DPO diagnostics in stage1 by default, disabled by `--disable_dpo_diagnostics`

Current code does **not** yet support the requested manifest-task interface:

```text
--preference_manifest selected_primary_comp.repaired.jsonl
--mask_from_manifest true
--train_mask_mode full
--train_mask_mode partial
--loss_region_mode full
--loss_region_mode region
--enable_dpo_diag true
```

## Required Code Gaps

Do not change loss math yet. The first required change is a manifest dataset
adapter that can read D2 repaired manifests.

Likely files to modify later:

```text
training/dpo/dataset/factory.py
training/dpo/dataset/generated_loser_manifest_dataset.py
training/dpo/train_stage1.py
training/dpo/train_stage2.py
training/dpo/scripts/03_dpo_stage1.sbatch
training/dpo/scripts/03_dpo_stage2.sbatch
training/dpo/scripts/run_stage1.py
training/dpo/scripts/run_stage2.py
```

Experiment-specific behavior:

- Experiment 5: load winner from `win_video_path`, loser from `final_loser_video_path`, ignore `mask_path`, feed full training mask.
- Experiment 6: same as experiment 5, but manifest `final_loser_video_path` must equal `raw_loser_video_path`.
- Experiment 7: load `mask_path`, set `M_train = M_gen`, feed partial training mask, keep full-video DPO loss.
- Experiment 8: same as experiment 7, but add region-weighted loss after experiment 7 is validated.

## Current Recommendation

Implement and smoke-test the manifest dataset adapter before touching
`compute_dpo_loss`. Experiment 5/6/7 can use the existing full-video DPO loss
once the dataset emits the expected tensors:

```text
pixel_values_pos
pixel_values_neg
conditioning_pixel_values
masks
input_ids
```
