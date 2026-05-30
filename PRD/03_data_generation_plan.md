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

Target experiment: `official_videodpo_diffueraser_data_partialmask_loser_comp_k4`

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

Target experiment: `official_videodpo_diffueraser_data_partialmask_loser_nocomp_k4`

```text
win = VideoDPO winner
partial_mask = used only for offline loser generation
raw_loser = video_inpainting_model(win, partial_mask)
final_loser = raw_loser
```

This is a diagnostic ablation. Mask-outside differences may appear and should be measured.

## Fixed V1 Policies

Do not split mask size, mask motion, mask position, or generation model choice
into separate first-round experiments. They are fixed data generation policies:

- mask policy: `videodpo_partialmask_policy_v1_medium_hard_k4`
- selection policy: `medium_hard_balanced_selection_v1`

Configs:

- `configs/generation/videodpo_partialmask_policy_v1_medium_hard_k4.yaml`
- `configs/generation/medium_hard_balanced_selection_v1.yaml`

The partial-mask policy creates K=4 interior-constrained irregular polygon masks
per VideoDPO winner. The selection policy scores all model/mask candidates and
selects primary/secondary medium-hard losers with equal source weights for
DiffuEraser, ProPainter, CoCoCo, and MiniMax-Remover.

Every generated candidate must be retained in `candidates_all.jsonl` before
selection. The comp and no-comp manifests must share the same selected candidate;
only `final_loser_video_path` differs.

## Model Integration Status

| Model | Code | Weights | Env | Status |
| --- | --- | --- | --- | --- |
| DiffuEraser | found | found, including PCM | found | canonical full/partial one-sample smoke OK |
| ProPainter | found | found | found | canonical full/partial one-sample smoke OK |
| CoCoCo | wrapper found | found, including SD inpaint root | found | canonical full/partial one-sample smoke OK |
| MiniMax-Remover | wrapper/cache found | found | found | canonical full/partial one-sample smoke OK |

## Generation Readiness Gate

As of the 2026-05-24 PAI probe:

- data roots, weight roots, generated-loser roots, and manifest schema scaffolds are prepared;
- YouTube-VOS frames and annotations are confirmed under the train split;
- four-model inference scripts compile and weight paths list successfully;
- real one-sample full-mask and partial-mask generation smoke passed for DiffuEraser, ProPainter, CoCoCo, and MiniMax-Remover;
- smoke outputs decode to 16 frames at 320x512, matching the canonical VideoDPO setting;
- partial-mask comp checks report outside-mask max absolute diff `0.000000`;
- `tools/offline_loser_generation.py` is still a planning/manifest scaffold and intentionally does not dispatch real inference.

Therefore the one-sample asset gate is passed. The next step is an explicit
calibration subset with disk estimate, selected model set, sample range, output
root, cheap metric scoring, selection, and manifest validation. Do not start
DPO training from this step.

Canonical smoke command:

```bash
python tools/pai_videodpo_single_sample_generation_smoke.py \
  --models all \
  --mask_modes full,partial \
  --output_root outputs/asset_smoke_tests/videodpo_single_sample \
  --run_generation
```

This command writes:

- `PRD/videodpo_canonical_data_setting.md`
- `outputs/asset_smoke_tests/videodpo_single_sample/report.md`
- `outputs/asset_smoke_tests/videodpo_single_sample/smoke_manifest.jsonl`

Without `--run_generation`, the same tool only prepares canonical inputs and
writes the setting report; that is useful for debugging but does not pass the
asset readiness gate.

Parallel four-model smoke command:

```bash
bash scripts/pai_run_parallel_generation_smokes.sh
```

The parallel wrapper runs one model per process/GPU, prints each model's result
table and failure log tail, and still generates only one canonical sample.

Passing smoke evidence:

| Model | Report |
| --- | --- |
| DiffuEraser | `outputs/asset_smoke_tests/parallel_generation_smoke_20260524_085008/diffueraser/report.md` |
| ProPainter | `outputs/asset_smoke_tests/parallel_generation_smoke_20260524_063024/propainter/report.md` |
| CoCoCo | `outputs/asset_smoke_tests/parallel_generation_smoke_20260524_070827/cococo/report.md` |
| MiniMax-Remover | `outputs/asset_smoke_tests/parallel_generation_smoke_20260524_070018/minimax_remover/report.md` |

## Calibration Before Full Generation

Before full generation, run a calibration subset with `--limit 20` or
`--limit 50`, all four models, and K=4 partial masks. The calibration must save
all candidates, score cheap metrics, select primary/secondary, and write:

- `PRD/generated_loser_calibration_report.md`
- `manifests/candidates_all.jsonl`
- `manifests/candidates_all.scored.jsonl`
- `manifests/selected_primary_comp.jsonl`
- `manifests/selected_primary_nocomp.jsonl`
- `manifests/selected_secondary_comp.jsonl`
- `manifests/selected_secondary_nocomp.jsonl`

Full generation can start only after this report shows acceptable fail rates,
reasonable `too_bad / eligible / too_good` ratios, no source-model selection
collapse, and comp outside-mask diff still equal or very close to zero.

PAI calibration entrypoint:

```bash
python tools/videodpo_generated_loser_calibration.py \
  --output_root data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4 \
  --models all \
  --limit 20 \
  --mask_policy_config configs/generation/videodpo_partialmask_policy_v1_medium_hard_k4.yaml \
  --selection_config configs/generation/medium_hard_balanced_selection_v1.yaml \
  --calibration_report PRD/generated_loser_calibration_report.md
```

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
