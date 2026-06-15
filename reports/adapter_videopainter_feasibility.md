# VideoPainter Adapter Feasibility

Date: 2026-06-15

Status: feasibility only. No training launched.

## Sources Checked

Local clone:

```text
/home/hj/dpo-2-1-exp/third_party_baselines/VideoPainter
```

GitHub:

```text
https://github.com/TencentARC/VideoPainter
```

## Evidence

The public GitHub repository exposes `train/`, `evaluate/`, and `infer/` folders and describes VideoPainter as official SIGGRAPH 2025 code for any-length video inpainting/editing.

The README includes a `Running Scripts / Training` section. The public training command uses CogVideoX-5B-I2V as the base model and launches:

```text
train_cogvideox_inpainting_i2v_video.py
```

Local training entrypoints found:

```text
train/VideoPainter.sh
train/VideoPainterID.sh
train/train_cogvideox_inpainting_i2v_video.py
train/train_cogvideox_inpainting_i2v_video_resample.py
```

Local `VideoPainter.sh` uses:

```text
MODEL_PATH=../ckpt/CogVideoX-5b-I2V
DATASET_PATH=../data/videovo/raw_video
meta_file_path=../data/pexels_videovo_train_dataset.csv
height=480
width=720
max_num_frames=49
inpainting_loss_weight=1.0
```

## Feasibility Checklist

| Question | Answer | Notes |
|---|---|---|
| Open source? | yes | TencentARC public repo. |
| Training code? | yes | `train/VideoPainter.sh` and Python train scripts. |
| Pretrained weights? | yes | README points to VideoPainter checkpoints and CogVideoX-5B-I2V. |
| Diffusion / DiT based? | yes | CogVideoX / video DiT family. |
| Can define policy/reference? | likely | Needs duplicated model/reference forward inside VideoPainter training. |
| Can train on YouTubeVOS? | possible after conversion | Our YouTubeVOS/D3 manifest must be converted to VideoPainter CSV/raw-video format. |
| Can eval on DAVIS? | yes with wrapper work | Need to save frame outputs and use our raw6 hard-comp metric wrapper. |
| Direct Diff-DPO fit? | possible but not plug-and-play | Need to inspect noise prediction tensors and integrate reference forward. |
| Output-level preference adapter? | possible fallback | Easier but weaker methodologically. |
| Cost | high | CogVideoX-5B and 49-frame settings are heavy. |

## Conclusion

Class: **B. needs modification before training**.

Exp14 follow-up audit refined the label to:

```text
direct_diff_dpo_design_feasible_not_implemented
```

VideoPainter is the only current adapter candidate with verified local training code and a diffusion/DiT architecture. It is feasible as a future isolated adapter gate, but not safe to launch directly from the current DiffuEraser scripts.

Required gate before training:

1. Create a new folder such as `exp14_adapter_videopainter/`.
2. Copy VideoPainter code into an isolated external-code area or submodule-like folder.
3. Convert our YouTubeVOS/D3 training manifest into VideoPainter's CSV/raw-video format.
4. Run a 1-5 step smoke test with the original VideoPainter objective.
5. Inspect tensors to decide direct Diff-DPO vs output-level preference adapter.
6. Keep evaluation fixed to raw6, hard comp, D+G off, frame-wise metrics.

Do not launch training until the user explicitly approves a VideoPainter adapter gate.

## Exp14 Status

Exp14 folder:

```text
exp14_adapter_videopainter/
experiment_registry/exp14_adapter_videopainter/
```

Smoke status:

- Smoke1: not run.
- Smoke20: not run.
- Gate2000: not ready.

Reason:

The upstream VideoPainter loop exposes the right diffusion tensors, but the
DPO adapter trainer has not been implemented yet. Do not start 2000-step
training until an isolated trainer exists and Smoke1 / Smoke20 pass on PAI.
