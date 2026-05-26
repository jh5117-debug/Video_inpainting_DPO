# Current Status

## Completed And Protected

### DiffuEraser Reproduction / SFT / Metric Setting

Status: completed and preserved.

Current best DiffuEraser evaluation setting:

- Denoise steps: `6`
- PCM acceleration: off
- Final mask dilation / compositing Gaussian blur: off
- Metric transfer: frame-wise, not mp4, because mp4 transport can reduce visual quality.

Local code references:

- `diffueraser_reproduction_sft/`
- `diffueraser/`
- `training/sft/`
- `tools/generate_diffueraser_fullmask_vbench.py`

### official-VideoDPO VC2

Status: completed qualitative and quantitative evaluation.

Recorded final score:

| Model | Total | Quality | Semantic | MeanRaw |
| --- | ---: | ---: | ---: | ---: |
| Official VC2 VideoDPO step3000 | 80.5997 | 82.8055 | 71.7763 | 0.6596 |

Recorded PAI artifacts are listed in `pai_audit_current_state.md`. They were not directly mounted in the current `hal-9000` audit session.

### official-VideoDPO DiffuEraser

Status: completed qualitative and quantitative evaluation.

This is the minimal-change model ablation: official VideoDPO / VC2 training skeleton was kept, while the model was replaced by the DiffuEraser full-mask bridge.

Recorded final VBench:

| Model | Total | Quality | Semantic | MeanRaw |
| --- | ---: | ---: | ---: | ---: |
| DiffuEraser-Base-Fullmask | 64.6162 | 74.4651 | 25.2204 | 0.3935 |
| DiffuEraser-Stage2-Fullmask | 73.6463 | 78.4804 | 54.3099 | 0.5560 |
| Delta | +9.0301 | +4.0153 | +29.0894 | +0.1625 |

Paper-style dimensions:

| Model | Total | Motion smooth. | Dynamic degree | Aesthetic quality | Object class | Multiple objects | Human action | Spatial relation. | Scene | Appear. style | Subject consist. | Back. consist. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DiffuEraser Base Fullmask | 64.62 | 98.33 | 0.28 | 36.28 | 14.18 | 2.15 | 6.20 | 9.14 | 0.47 | 22.78 | 99.44 | 99.09 |
| DiffuEraser Stage2 Fullmask | 73.65 | 97.30 | 44.72 | 51.77 | 69.08 | 24.59 | 66.20 | 26.03 | 27.49 | 23.79 | 95.87 | 98.34 |
| Delta | +9.03 | -1.03 | +44.44 | +15.49 | +54.91 | +22.44 | +60.00 | +16.89 | +27.02 | +1.01 | -3.57 | -0.76 |

## Current Open Problems

- PAI NAS paths were not mounted in this local audit session; destructive cleanup must be done only on the PAI node after confirming `/mnt/nas` / `/mnt/workspace`.
- Four generation models have passed one-sample smoke, but the active 2026-05-25 production data path is now DiffuEraser-only because four-model generation is too slow for full data creation.
- D1 fullmask OR generation was paused after visual artifacts. The old H20-2 D1 root used the DiffuEraser OR stack (`inference/run_OR.py` -> `diffueraser_OR.py`), not the BR stack (`diffueraser.py`). Full-frame masks leave OR with no visible background context, so generated samples became abstract/blurry and are invalid as meaningful fullmask losers. Do not train from the old OR-fullmask root. It is retained only as failure-audit evidence.
- D1 BR/no-prior validation technically passed manifest/decode checks, but failed the quality gate at limit=100: 95/100 rows were `too_bad`, median quality score was `0.1947`, and max quality score was only `0.3315`. Do not start full D1 generation or train experiment 4 from this data unless experiment 4 is explicitly reframed as a diagnostic/failure-case ablation.
- The partial-mask K=4 generated-loser run should use `MODELS=diffueraser` first. Each VideoDPO winner produces four DiffuEraser candidates, one per mask; selection still writes primary/secondary manifests, but production training should consume only the selected primary manifest unless a later ablation says otherwise.
- PAI throughput tuning is still active. The failed high-concurrency probe showed host overload (`load average` above 3000, GPU util near 0, GPU memory occupied), so do not use very high `WORKERS_PER_GPU` just to fill memory. Use the guarded launcher thread limits and tune from `WORKERS_PER_GPU=4`, `SHARD_SIZE=1`.
- New ablations should be introduced as separate experiment directories and should reuse existing training/model code.

## Active Production Data Runs On 2026-05-25

### D1 / Experiment 4: Fullmask DiffuEraser-Only Loser Data

Status: old OR-fullmask invalid; BR/no-prior smoke and limit=100 technical gates passed, quality gate failed.

Old OR output root, retained only for failure audit:

```text
/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_fullmask_loser
```

BR/no-prior validation root:

```text
/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_fullmask_loser_br_noise
```

Important subpaths:

- Old OR shards: `/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_fullmask_loser/_shards`
- BR/no-prior shards: `/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_fullmask_loser_br_noise/_shards`
- BR/no-prior manifests: `/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_fullmask_loser_br_noise/manifests`
- BR/no-prior reports: `/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_fullmask_loser_br_noise/reports`

Run identity:

```text
experiment = official_videodpo_diffueraser_data_fullmask_loser
data_id = D1
machine = H20-2
generation_source = diffueraser_only
generation_model = diffueraser
diffueraser_inference_stack = br
diffueraser_prior_mode = noise
mask_mode = full
num_masks_per_video = 1
comp = false
final_loser = raw_loser
process_name = lingbot-world
```

Old OR run observed status before pause:

```text
done = 1687 / 10000
failed = 4
```

Later spot-check at `2026-05-25` confirmed the active D1 logs call:

```text
inference/run_OR.py
from diffueraser.diffueraser_OR import DiffuEraser
Priori generating: ...
```

This means the old D1 fullmask output is **OR fullmask**, not BR fullmask.
Visual samples showed severe blur/abstract artifacts. Treat the old D1 artifacts
as suspect. D2 partialmask remains less risky because OR receives local masks
and the comp manifest preserves pixels outside the generated mask.

BR/no-prior validation result:

```text
limit20 rows = 20
limit100 rows = 100
status = OK: 100
generation_source = diffueraser_only: 100
diffueraser_inference_stack = br: 100
diffueraser_prior_mode = noise: 100
propainter.mp4 count = 0
diffueraser.mp4 count = 100
selected_primary_fullmask.jsonl = 100
selected_secondary_fullmask.jsonl = 100
frame / resolution audit = pass, 16 frames at 512x320 storage / canonical 320x512
quality buckets = too_bad: 95, texture_or_structure_shift: 5
quality_score median = 0.1947
quality_score mean = 0.1884
quality_score max = 0.3315
```

Preview paths:

- `/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_fullmask_loser_br_noise/reports/previews/d1_br_noise_limit20_win_raw_first_frame.jpg`
- `/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_fullmask_loser_br_noise/reports/previews/d1_br_noise_limit100_win_raw_first_frame.jpg`

Decision: D1 BR/no-prior is technically runnable but not accepted as a main
training data source. Experiment 4 should be downgraded to diagnostic/failure
case unless a new D1 loser definition is introduced.

Interpretation note: `generation_source=diffueraser_only` means only the
DiffuEraser candidate is written as the manifest source and selected loser. It
does not mean all DiffuEraser paths use the same prior policy. For D2
partialmask, the active path is still OR + ProPainter prior. For new D1
fullmask validation, the required path is BR + `prior_mode=noise`, so no
ProPainter prior is generated or passed.

### D2 / Experiments 5, 6, 7, 8: Partialmask K4 DiffuEraser-Only Loser Data

Status: generated on PAI; post-generation audit/repair completed; final
training-readiness check pending before any long training.

Output root:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4
```

Run identity:

```text
data_id = D2
generation_source = diffueraser_only
generation_model = diffueraser
diffueraser_inference_stack = or
diffueraser_prior_mode = propainter
mask_mode = partial
num_masks_per_video = 4
comp manifest serves experiment 5
nocomp manifest serves experiment 6
comp data + mask serves experiment 7
comp data + mask also serves experiment 8 region-loss training
```

Final observed status from PAI at `2026-05-26`:

```text
done_shards = 10000 / 10000
failed_shards = 0
candidate_rows = 40000 / 40000
status = OK: 40000
generation_model = diffueraser: 40000
candidates_all.jsonl = 40000
candidates_all.scored.jsonl = 40000
selected_primary_comp.jsonl = 10000
selected_primary_nocomp.jsonl = 10000
selected_secondary_comp.jsonl = 10000
selected_secondary_nocomp.jsonl = 10000
selection_events.jsonl = 10000
```

The raw D2 manifests were generated before the explicit metadata fields were
added. The repair step has been run and confirmed:

```text
generation_source = diffueraser_only
diffueraser_inference_stack = or
diffueraser_prior_mode = propainter
```

Repaired manifests:

```text
manifests/candidates_all.repaired.jsonl
manifests/candidates_all.scored.repaired.jsonl
manifests/selected_primary_comp.repaired.jsonl
manifests/selected_primary_nocomp.repaired.jsonl
manifests/selected_secondary_comp.repaired.jsonl
manifests/selected_secondary_nocomp.repaired.jsonl
```

Run the final training-readiness check before implementing or launching
experiments 5/6/7/8:

```bash
OUT=/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4
python tools/d2_training_readiness_check.py --output_root "$OUT"
```

This writes:

```text
reports/d2_training_readiness_report.md
```
