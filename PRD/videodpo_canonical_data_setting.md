# VideoDPO Canonical Data Setting

This checked-in file is a template. The archived 2026-05-24 all-model asset
smoke overwrote it with:

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

The 2026-05-24 PAI smoke runs used the first real VideoDPO winner sample from
the PAI train YAML:

- pair_index: `0`
- winner_video_path: `/mnt/nas/hj/data/external/hf/vidpro10k-vc2-dataset/_extracted/home/liurt/liurt_data/haoyu/dataset/vidpro10k-vc2-dataset/winvideos/000000.mp4`
- prompt source: VideoDPO pair metadata loaded from `VIDEO_DPO_TRAIN_DATA_YAML`

Canonical full-mask and partial-mask one-sample smoke passed for all four
generation models, but current production data generation is DiffuEraser-only:
`generation_source=diffueraser_only`.
