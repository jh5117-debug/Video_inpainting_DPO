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
- D1 fullmask OR generation was paused after visual artifacts. The old H20-2 D1 root used the DiffuEraser OR stack (`inference/run_OR.py` -> `diffueraser_OR.py`), not the BR stack (`diffueraser.py`). Full-frame masks leave OR with no visible background context, so generated samples can become abstract/blurry and may be invalid as meaningful fullmask losers. Do not train from the old OR-fullmask root. New D1 validation must use the BR/no-prior path (`DIFFUERASER_INFERENCE_STACK=br`, `DIFFUERASER_PRIOR_MODE=noise`) in a separate output root.
- The partial-mask K=4 generated-loser run should use `MODELS=diffueraser` first. Each VideoDPO winner produces four DiffuEraser candidates, one per mask; selection still writes primary/secondary manifests, but production training should consume only the selected primary manifest unless a later ablation says otherwise.
- PAI throughput tuning is still active. The failed high-concurrency probe showed host overload (`load average` above 3000, GPU util near 0, GPU memory occupied), so do not use very high `WORKERS_PER_GPU` just to fill memory. Use the guarded launcher thread limits and tune from `WORKERS_PER_GPU=4`, `SHARD_SIZE=1`.
- New ablations should be introduced as separate experiment directories and should reuse existing training/model code.

## Active Production Data Runs On 2026-05-25

### D1 / Experiment 4: Fullmask DiffuEraser-Only Loser Data

Status: old OR-fullmask run paused; BR/no-prior replacement requires small-sample validation.

Old OR output root, retained only for failure audit:

```text
/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_fullmask_loser
```

New BR/no-prior validation root:

```text
/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_fullmask_loser_br_noise
```

Important subpaths:

- Old OR shards: `/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_fullmask_loser/_shards`
- New BR/no-prior shards: `/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_fullmask_loser_br_noise/_shards`
- New BR/no-prior manifests after launcher merge: `/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_fullmask_loser_br_noise/manifests`
- New BR/no-prior reports after launcher merge: `/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_fullmask_loser_br_noise/reports`

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

Interpretation note: `generation_source=diffueraser_only` means only the
DiffuEraser candidate is written as the manifest source and selected loser. It
does not mean all DiffuEraser paths use the same prior policy. For D2
partialmask, the active path is still OR + ProPainter prior. For new D1
fullmask validation, the required path is BR + `prior_mode=noise`, so no
ProPainter prior is generated or passed.

### D2 / Experiments 5, 6, 7: Partialmask K4 DiffuEraser-Only Loser Data

Status: running on PAI.

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
```

Current observed status from PAI at `2026-05-25`:

```text
done_shards = 2052 / 10000
failed_shards = 0
candidate_rows = 8228 / 40000
status = OK: 8228
rate = 461.3 shards/hour
estimated finish = 2026-05-26 05:50 CST
```

The PAI monitoring variables must use full-run values:

```bash
export TOTAL_PAIRS=10000
export SHARD_SIZE=1
export EXPECT_ROWS=$((TOTAL_PAIRS * 4))
```

If `TOTAL_PAIRS=100` is left from a validation run, progress prints such as
`done_shards=2052 / 100` are misleading. The existing merged selected manifests
with 100 rows are historical validation-run artifacts until the full launcher
finishes and merges all shards again.
