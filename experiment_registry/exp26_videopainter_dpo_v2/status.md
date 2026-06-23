# Exp26 VideoPainter DPO v2 Status

- branch: `research/exp26-videopainter-dpo-v2`
- worktree: `/home/hj/H20_Video_inpainting_DPO_exp26_videopainter`
- status: `VP2_STATIC_FIXES_STARTED`
- copied source: `exp14_adapter_videopainter/`
- current scope: static v2 trainer fixes and unit tests
- GPU work: not started
- DAVIS50: not started

Initial fixes:

- official optimizer fields exposed;
- `noised_image_dropout` wired into image latent preparation;
- first-frame consistency helper added;
- native 49-frame policy enforced, 13F requires plumbing flag;
- formal mode now rejects 16-frame inputs instead of trimming them to 13;
- `--first_frame_gt` / `--no-first_frame_gt` now controls first-frame
  consistency;
- official optimizer/scheduler parser added for locked parity config;
- `itertools.cycle(loader)` removed;
- loser-dominant definition aligned with project diagnostics;
- strict checkpoint reload helper added.
# Exp26 Registry Status

L0_L4_PASSED
FORMAL_49F_SPLIT_BLOCKED

## 49F Source Diagnostic

- Source root: `/mnt/workspace/hj/nas_hj/data/external/ytbv_2019_full_resolution/train`
- Required valid candidates: 640
- Valid candidates: 0
- Failed candidates: 3471
- Max frame count seen: 36
- Max mask count seen: 36
- Gate64 official baseline self-loser generation: not launched.

Reason: active YouTube-VOS source is a sparse extraction and does not satisfy formal 49-frame input requirements.
