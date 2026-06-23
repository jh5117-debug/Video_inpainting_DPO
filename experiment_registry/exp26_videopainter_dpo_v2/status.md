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
- `itertools.cycle(loader)` removed;
- loser-dominant definition aligned with project diagnostics;
- strict checkpoint reload helper added.
