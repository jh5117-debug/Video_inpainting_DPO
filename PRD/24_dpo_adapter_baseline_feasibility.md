# PRD 24: Adapter Baseline Feasibility

Date: 2026-06-15

## Current Scope

This is a feasibility plan only. No adapter training has been launched.

The adapter scope is intentionally narrowed to two candidates:

1. `VideoPainter`
2. `MiniMax-Remover`

All other baselines are ignored for the current adapter phase. This does not mean they are unimportant; it only means they are out of scope for the next adapter gate.

## Out Of Scope For This Phase

Do not pursue adapter training for:

- `FFF-VDI`
- `ProPainter`
- `E2FGVI`
- `STTN`
- `FloED`
- `CoCoCo`
- `VACE`
- `LGVI`
- `RT-Remover`
- `VideoComposer` / `VideoComp`

Reasons:

- non-diffusion models such as ProPainter / E2FGVI / STTN are not direct Diff-DPO adapter targets;
- several diffusion baselines are inference-only, training-free, or not validated as trainable in our environment;
- the next adapter step should stay small and isolated.

## VideoPainter Feasibility

Report:

```text
reports/adapter_videopainter_feasibility.md
```

Conclusion: **B. needs modification before training**.

Why:

- VideoPainter is open source.
- It exposes training scripts and CogVideoX inpainting training entrypoints.
- A local clone exists at:

```text
/home/hj/dpo-2-1-exp/third_party_baselines/VideoPainter
```

- Local training entrypoints include:

```text
train/VideoPainter.sh
train/VideoPainterID.sh
train/train_cogvideox_inpainting_i2v_video.py
train/train_cogvideox_inpainting_i2v_video_resample.py
```

But it is not plug-and-play for our DPO objective:

- data format must be converted from our YouTube-VOS/D3 manifest into VideoPainter CSV/raw-video format;
- policy/reference forward passes must be inserted around its CogVideoX/branch training tensors;
- our fixed DAVIS / YouTubeVOS raw6 hard-comp evaluation wrapper must be connected to its frame outputs;
- likely compute cost is high.

Recommendation: prepare a small isolated smoke gate only after explicit confirmation. Do not launch now.

## MiniMax-Remover Feasibility

Report:

```text
reports/adapter_minimax_remover_feasibility.md
```

Conclusion: **C. frozen baseline only for now**.

Why:

- MiniMax-Remover is open source and diffusion/DiT-style.
- The public repo exposes inference pipeline / transformer / gradio demo / test script and model weight download.
- The current local project only has an inference wrapper:

```text
DPO_finetune/infer_minimax_candidate.py
weights/minimax_remover/.gitkeep
```

- No adapter-ready training script or training data path has been validated locally.

Recommendation: do not claim MiniMax-Remover is trainable for DPO adapter unless official train scripts and training data are found and verified. For now it can be used only as a frozen baseline or related-work comparison.

## Adapter Decision Table

| Candidate | Open source | Training code verified | Diffusion / DiT | Can train on our YouTubeVOS now | Direct Diff-DPO fit | Current class | Action |
|---|---:|---:|---:|---:|---:|---|---|
| VideoPainter | yes | yes | yes | needs conversion | possible after code integration | B: needs modification | prepare adapter gate only after confirmation |
| MiniMax-Remover | yes | no | yes | no | not now | C: frozen baseline | no training; frozen baseline only |

## Do Not Launch Training

No adapter training should start until the user explicitly says to start a specific adapter gate.

If a gate is approved, each adapter must have:

- independent experiment folder
- independent registry
- copied external code isolated from existing Exp11 / Exp12 folders
- fixed inpainting eval protocol
- dpo/adapter diagnostics
- four-column visualizations
