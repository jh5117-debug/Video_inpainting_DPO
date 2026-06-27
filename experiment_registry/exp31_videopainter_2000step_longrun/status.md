# Exp31 VideoPainter 2000-Step Long-Run Status

Current status: `VIDEOPAINTER_2000_STEP0_50_2000_EVAL_RUNNING`

- branch: `research/exp31-videopainter-2000step-longrun-20260627`
- base: `origin/research/exp26-videopainter-dpo-v2`
- base HEAD: `568a7dfb48bcdfce893176a1dd48c653414a13a8`
- worktree: `/home/hj/H20_Video_inpainting_DPO_exp31_vp2000`
- GPU plan: GPU1, with GPU0 and GPU5 reserved for right-side or stale-lock
  protection.
- resume policy: fresh total-2000 from Step0 because Step50 lacks scheduler and
  RNG state.
- L0/L1: passed on GPU1 in run `exp31_vp_l0_l1_20260627_132158`.
- checkpoint ladder: ready, with explicit protected checkpoints
  `0,1,10,50,100,200,500,1000,1500,2000`.
- trainer state: optimizer, explicit `lr_scheduler`, and RNG state are saved.
- training: completed in run `exp31_vp2000_fresh_step0_20260627_133831`.
- final training step: `2000`.
- evaluation: running in
  `exp31_vp2000_eval_step0_50_2000_20260628_032700`.
- evaluation GPU: GPU1.
- evaluation checkpoints: `step0`, `step50`, `step2000`.
- evaluation splits: fixed search-dev and fixed shadow-dev.
- next milestone: complete Step0/50/2000 metrics and visual review.

No Exp26 or Exp30 files were modified.
