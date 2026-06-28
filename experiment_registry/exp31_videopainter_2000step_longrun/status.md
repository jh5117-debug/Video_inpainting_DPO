# Exp31 VideoPainter 2000-Step Long-Run Status

Current status: `VIDEOPAINTER_2000_PARETO_MIXED`

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
- evaluation: completed in
  `exp31_vp2000_eval_step0_50_2000_20260628_032700`.
- evaluation GPU: GPU1.
- evaluation checkpoints: `step0`, `step50`, `step2000`.
- evaluation splits: fixed search-dev and fixed shadow-dev.
- search-dev status: `VIDEOPAINTER_2000_SEARCHDEV_EVALUATED`.
- shadow-dev status: `VIDEOPAINTER_2000_SHADOWDEV_EVALUATED`.
- external status: `VIDEOPAINTER_2000_EXTERNAL_NOT_AVAILABLE`.
- visual review: completed from all-32 evidence and crop pages for Step0,
  Step50, and Step2000 on both splits.
- final decision: `VIDEOPAINTER_2000_PARETO_MIXED`.
- reason: available metrics and video evidence strongly favor Step2000, but
  LPIPS and Ewarp were not computed in this fast summary, so the formal
  `VIDEOPAINTER_2000_POSITIVE` gate is not satisfied.
- strict readback: `VIDEOPAINTER_2000_STRICT_READBACK_COMPLETE_BASE_AUDIT_PENDING`.
- strict readback report: `reports/exp31_vp_2000_strict_readback.md`.

No Exp26 or Exp30 files were modified.
