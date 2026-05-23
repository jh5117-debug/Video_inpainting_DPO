# Data Generation Plan

## Manifest Schema

Every offline generation run must save a manifest with:

```text
sample_id
prompt
win_video_path
raw_loser_video_path
comp_loser_video_path
final_loser_video_path
mask_path
mask_mode
mask_convention
comp
generation_model
source_dataset
seed
fps
num_frames
height
width
```

Keep this schema compatible across fullmask, partialmask comp, partialmask no-comp, and future YouTube-VOS data.

## Full-Mask Loser Generation

Target experiment: `official_videodpo_diffueraser_data_fullmask_loser`

```text
win = VideoDPO winner
full_mask = all masked
raw_loser = video_inpainting_model(win, full_mask)
final_loser = raw_loser
```

Training still uses official VideoDPO / DiffuEraser full-mask bridge.

### PAI Data Source

Use the same VideoDPO PAI train-data YAML as the completed official VC2 and
official DiffuEraser experiments:

```bash
VIDEO_DPO_DATA_ROOT=/mnt/nas/hj/data/VideoDPO
VIDEO_DPO_TRAIN_DATA_YAML=/mnt/nas/hj/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.pai.yaml
```

The YAML points to the extracted VidPro10K/VC2 root and the completed training
logs confirm `DPO dataset has 10000 pairs`. New generated-loser manifests
should preserve compatibility with this pair order and sample identity.

### Mask Convention Audit

Do not infer black/white mask semantics from experiment names. Confirm them per model.

For the current DiffuEraser / VideoDPO bridge, local code says:

- `training/dpo/dataset/videodpo_fullmask_dataset.py`: the full-hole training bridge uses a black conditioning image, and `videodpo_full_mask_value=0.0` means unknown/hole in the BrushNet mask channel.
- `tools/generate_diffueraser_fullmask_vbench.py`: `--full_mask_value 0.0 --mask_value_space internal` maps to PIL mask pixel `255`, because DiffuEraser preprocessing maps white PIL masks to internal `0/hole`.

Therefore for DiffuEraser fullmask generation, use internal mask value `0.0` for full-hole/full-frame generation. For ProPainter, CoCoCo, and MiniMax-Remover, audit their own mask convention before running; do not assume DiffuEraser's PIL white/internal zero convention applies.

## Partial-Mask + Comp

Target experiment: `official_videodpo_diffueraser_data_partialmask_loser_comp`

```text
win = VideoDPO winner
partial_mask = used only for offline loser generation
raw_loser = video_inpainting_model(win, partial_mask)
comp_loser = win * (1 - partial_mask) + raw_loser * partial_mask
final_loser = comp_loser
```

Training still uses full-mask bridge. Partial mask is not passed to the model during training. This is the cleanest data-only partial-mask ablation.

The formula above is semantic. Real implementation must first normalize mask polarity so that:

- mask outside region comes exactly from `win`;
- mask inside region comes from `raw_loser`;
- saved metadata records original mask convention and any inversion applied.

## Partial-Mask + No-Comp

Target experiment: `official_videodpo_diffueraser_data_partialmask_loser_nocomp`

```text
win = VideoDPO winner
partial_mask = used only for offline loser generation
raw_loser = video_inpainting_model(win, partial_mask)
final_loser = raw_loser
```

This is a diagnostic ablation. Mask-outside differences may appear and should be measured.

## Model Integration Status

| Model | Code | Weights | Env | Status |
| --- | --- | --- | --- | --- |
| DiffuEraser | found | found | found | path-ready; real generation smoke not run |
| ProPainter | found | found | found | path-ready; real generation smoke not run |
| CoCoCo | wrapper found | found | found | path-ready; real generation smoke not run |
| MiniMax-Remover | wrapper/cache found | found | found | path-ready; real generation smoke not run |

## Generation Readiness Gate

As of the 2026-05-24 PAI probe:

- data roots, weight roots, generated-loser roots, and manifest schema scaffolds are prepared;
- YouTube-VOS frames and annotations are confirmed under the train split;
- four-model inference scripts compile and weight paths list successfully;
- real one-sample full-mask and partial-mask generation smoke has not run;
- `tools/offline_loser_generation.py` is still a planning/manifest scaffold and intentionally does not dispatch real inference.

Therefore the next step is not full data generation yet. Run one-sample smoke
per model first, then start offline generation only for models that pass video
decode, fps, frame-count, resolution, and comp outside-mask checks.

## Online Loser Generation

Online loser generation is future work. Do not start it until offline generation is stable and all four generator runtimes are confirmed.

## Data-Only vs Task Boundary

Experiments 1, 2A, and 2B are data-only:

- mask is used only during offline loser generation;
- DPO training still uses `official_videodpo_diffueraser` full-mask bridge;
- the model does not receive the partial mask during training;
- changed variable is data.

Experiment 3 is task-level:

- partial mask is both saved in data and passed into the model during training;
- DiffuEraser performs local partial video inpainting;
- changed variable is task.
