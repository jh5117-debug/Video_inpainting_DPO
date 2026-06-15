# Adapter Gate Candidate Recommendation

Status: prepared recommendation only. No training launched.

## Recommended First Gate

**VideoPainter adapter gate**

Why:

- It is video inpainting/editing oriented.
- It exposes a training script.
- It is diffusion/DiT based, which is closer to the current DiffuEraser DPO family than ProPainter/E2FGVI/STTN.
- Its training/eval code is more relevant to a future adapter objective than inference-only baselines.

Required before launch:

1. Create `experiment_registry/exp14_adapter_videopainter/`.
2. Clone/copy VideoPainter into an isolated external-code folder.
3. Do not mix its code into Exp11/Exp12 folders.
4. Verify its training script can run a 1-5 step smoke on YouTube-VOS.
5. Decide whether the objective is direct diffusion DPO or an output-level preference proxy after inspecting actual tensors.
6. Keep evaluation fixed to our raw6 hard-comp metric wrapper where possible.

Risk:

- Very high GPU/memory cost.
- Potential dependency conflicts with current DiffuEraser environment.
- The model is larger and may not fit a quick 2000-step experiment without simplifying.

## Backup Gate

**FFF-VDI adapter gate**

Why:

- It is diffusion-based and has `train.py`.
- It trains on YouTube-VOS.
- It is conceptually closer to direct diffusion DPO.

Risk:

- The README recommends 8+ GPUs with 80GB VRAM.
- The repo is young and training details are incomplete.

## Not Recommended For Direct Diff-DPO

- ProPainter, E2FGVI, STTN: trainable, but non-diffusion. Use only for output-level preference/fine-tune adapter.
- CoCoCo, FloED, VACE, MiniMax-Remover, LGVI, RT-Remover, VideoComp: no adapter-ready training path found or task mismatch.

## Launch Decision

Do not launch now. Finish:

1. YouTubeVOS100 + DAVIS50 extended eval.
2. Final 20 visual cases.
3. Then decide whether VideoPainter gate is worth the compute.
