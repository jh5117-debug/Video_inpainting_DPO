# Exp31 VideoPainter 2000-Step Long-Run Status

Current status: `VIDEOPAINTER_2000_L0_L1_PASSED`

- branch: `research/exp31-videopainter-2000step-longrun-20260627`
- base: `origin/research/exp26-videopainter-dpo-v2`
- base HEAD: `568a7dfb48bcdfce893176a1dd48c653414a13a8`
- worktree: `/home/hj/H20_Video_inpainting_DPO_exp31_vp2000`
- GPU plan: GPU1, with GPU0 and GPU5 reserved for right-side or stale-lock
  protection.
- resume policy: fresh total-2000 from Step0 because Step50 lacks scheduler and
  RNG state.
- L0/L1: passed on GPU1 in run `exp31_vp_l0_l1_20260627_132158`.
- training: not started.
- next milestone: fresh-from-Step0 2000-step training.

No Exp26 or Exp30 files were modified.
