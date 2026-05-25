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
- The partial-mask K=4 generated-loser run should use `MODELS=diffueraser` first. Each VideoDPO winner produces four DiffuEraser candidates, one per mask; selection still writes primary/secondary manifests, but production training should consume only the selected primary manifest unless a later ablation says otherwise.
- PAI throughput tuning is still active. The failed high-concurrency probe showed host overload (`load average` above 3000, GPU util near 0, GPU memory occupied), so do not use very high `WORKERS_PER_GPU` just to fill memory. Use the guarded launcher thread limits and tune from `WORKERS_PER_GPU=4`, `SHARD_SIZE=1`.
- New ablations should be introduced as separate experiment directories and should reuse existing training/model code.
