# VideoDPO Canonical Data Setting

This checked-in file is a template. On PAI it is overwritten by:

```bash
python tools/pai_videodpo_single_sample_generation_smoke.py --models all --mask_modes full,partial --run_generation
```

Expected canonical setting from the completed official DiffuEraser runs:

- train_data_yaml: `/mnt/nas/hj/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.pai.yaml`
- source config: `DPO_finetune/configs/official_diffueraser_stage1.yaml`
- canonical_height: `320`
- canonical_width: `512`
- canonical_num_frames: `16`
- canonical_frame_stride: `1`
- canonical_resize_policy: exact resize to `train_width` x `train_height`, matching the current `VideoDPOFullMaskDiffuEraserDataset` loader.
- canonical_crop_policy: none in the current dataset loader.
- canonical_full_mask_value: `0.0` in the training mask tensor.
- generator mask PNG value: `255` for the inpaint region.

Full data generation must wait until the PAI-generated version of this file
records a real winner sample, prompt, raw video info, frame indices, and smoke
status for the selected generation models.
