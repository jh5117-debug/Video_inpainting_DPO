# BR Adapter Candidate Training-Code Audit

Date: 2026-06-14

This audit checks whether the methods named by the user expose usable training
code for a future BR/inpainting adapter experiment. This does not launch any
training and does not modify current Exp9/10/11/12 code.

## Summary

| Method | Public repo / page | Training code status | Adapter action |
|---|---|---|---|
| VideoComp / VideoComposer | `https://github.com/ali-vilab/videocomposer` | No usable training entry found in public tree. Repo contains inference-oriented runner plus datasets/loss utilities, but no train/finetune script. | Do not start adapter. Treat as inference/reference only unless training code is later found. |
| COCOCO | `https://github.com/zibojia/COCOCO` | README says this is inference code and TODO says training code is under preparation. Public tree has validation/inference scripts, no train entry. | Do not start adapter. |
| FloED | arXiv page found, no official public GitHub/training repo found in current search. | No training code found. | Do not start adapter. |
| VideoPainter | `https://github.com/TencentARC/VideoPainter` | Training code is public. Repo includes `train/VideoPainter.sh`, `train/VideoPainterID.sh`, `train/train_cogvideox_inpainting_i2v_video.py`, and `train/train_cogvideox_inpainting_i2v_video_resample.py`. | Candidate for isolated BR adapter import after current Exp12 outer comparison finishes. |
| VACE | `https://github.com/ali-vilab/VACE` | Public tree exposes inference files (`vace_wan_inference.py`, `vace_ltx_inference.py`) and no train/finetune/dataset/loss entry. | Do not start adapter. |

## Evidence

- COCOCO project page links to GitHub, and GitHub README states it is inference
  code. Its TODO says training code is under preparation.
- VACE project page links to GitHub. Recursive tree check found inference files
  only.
- VideoComposer project page links to GitHub. Recursive tree check found
  `artist/ops/losses.py` and `tools/videocomposer/datasets.py`, but no training
  script.
- VideoPainter GitHub README has a `Training` section and explicit accelerate
  launch commands for `train_cogvideox_inpainting_i2v_video.py` and
  `train_cogvideox_inpainting_i2v_video_resample.py`.

## Current Decision

Do not copy or adapt any baseline code until the current Exp12 adaptive outer
boundary run is scored under the fixed DAVIS50 protocol.

If the user continues with BR adapter work, the only currently viable training
code import target is VideoPainter. It must be copied into an isolated folder
such as `third_party_br_adapters/videopainter_official/`, with our DPO training
and fixed DAVIS50 metric wrappers added outside the upstream source tree.
