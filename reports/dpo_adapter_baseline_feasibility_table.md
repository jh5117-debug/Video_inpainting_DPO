# DPO Adapter Baseline Feasibility

Status: feasibility only. No adapter training launched.

## Decision

The immediate adapter-gate candidates are:

1. **VideoPainter**: diffusion/DiT video inpainting with an exposed training script, but very high cost.
2. **FFF-VDI**: diffusion-based video inpainting with `train.py`, YouTube-VOS training path, and accelerate workflow, but also high cost.

The practical non-diffusion candidates are:

1. **ProPainter**
2. **E2FGVI**
3. **STTN**

These have training code, but they are not diffusion noise-prediction models, so they should not be called direct Diff-DPO adapters. They could support an output-level preference/fine-tune adapter if we define a model-specific objective.

Do **not** treat the following as trainable adapter candidates right now:

- MiniMax-Remover
- CoCoCo
- FloED
- VACE
- LGVI
- RT-Remover
- VideoComp

## Table

| Baseline | Category | Training Code | Diffusion | Adapter Feasibility | Note |
|---|---|---:|---:|---|---|
| ProPainter | B output-level | yes | no | output-level only | Official repo has `train.py`, YouTube-VOS/DAVIS setup, and ProPainter training configs. |
| E2FGVI | B output-level | yes | no | output-level only | Official repo has `train.py` and E2FGVI/HQ configs. |
| STTN | B output-level | yes | no | output-level only | Official repo has `train.py --config configs/youtube-vos.json`. |
| FloED | C frozen | no / training-free | yes | frozen baseline | Project describes a training-free flow-guided diffusion inference method. |
| FFF-VDI | A direct candidate | yes | yes | direct diffusion candidate | Repo has `train.py`, `dataset.py`, and YouTube-VOS training instructions; cost is high. |
| CoCoCo | C frozen | no | yes | frozen baseline | Repo is inference/training-free and says training code is under preparation. |
| VideoPainter | A direct candidate | yes | yes | direct diffusion candidate | Repo shows CogVideoX inpainting training command; high-cost gate only. |
| VACE | C frozen | no | yes | frozen baseline | README exposes inference/preprocess setup, not training. |
| MiniMax-Remover | C frozen | no train command found | yes | frozen baseline | Do not train unless official train scripts are verified later. |
| RT-Remover | D related | no public training found | likely | related work | Paper entry found, no adapter-ready code. |
| LGVI | C frozen | no train.py found | yes | frozen / custom task | Language-driven task, inference/checkpoints released; not our binary-mask adapter path. |
| VideoComp | D related | no | unclear | related work | Not a directly trainable BR inpainting baseline from current evidence. |

CSV: `reports/dpo_adapter_baseline_feasibility_table.csv`

## Source Notes

- ProPainter official repo lists training configs and `python train.py -c configs/train_propainter.json`.
- E2FGVI official repo lists `train.py` and training configs.
- STTN official repo lists `python train.py --config configs/youtube-vos.json --model sttn`.
- FFF-VDI official repo lists `train.py`, YouTube-VOS training, and recommends 8+ GPUs with 80GB VRAM.
- VideoPainter official repo includes a CogVideoX inpainting training command.
- CoCoCo repo says inference/training-free usage and that training code is under preparation.
- VACE repo release note and README focus on inference/preprocessing/gradio.

## Recommendation

Prepare an adapter gate for **VideoPainter** only after we finish the YouTubeVOS100 + DAVIS50 visual/metric evidence. It is the most directly aligned with a modern trainable video inpainting adapter path, but the risk/cost is high enough that it should not be launched without an explicit go/no-go.
