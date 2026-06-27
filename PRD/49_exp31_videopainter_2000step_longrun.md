# PRD 49: Exp31 VideoPainter 2000-Step Long-Run

Date: 2026-06-27

## Objective

Run an isolated VideoPainter 2000-step long-run so the VideoPainter evidence is
not limited to the Exp26 50-step micro gate.

## Isolation

- Branch: `research/exp31-videopainter-2000step-longrun-20260627`
- Base: `origin/research/exp26-videopainter-dpo-v2`
- Base HEAD: `568a7dfb48bcdfce893176a1dd48c653414a13a8`
- HAL worktree: `/home/hj/H20_Video_inpainting_DPO_exp31_vp2000`
- PAI runtime root: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp31_vp2000`
- Experiment output root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp31_videopainter_2000step_longrun`
- Log root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp31_videopainter_2000step_longrun`

Exp31 must not write to Exp26 outputs, Exp30 outputs, shared trainer code, or
`inference/metrics.py`.

## Source Readback

Exp26 fixed 50-step identity:

- run: `vp_primary32_50step_20260625_171032`
- primary32 SHA256:
  `82f661f2f30a581a213972533817624217eabb97eba7aaeedc00ee2109e4e716`
- search-dev SHA256:
  `41c6571d26e4a5130818dd50fbbe1314c9d953284561a3cd20f630572f7c2a71`
- shadow-dev SHA256:
  `0338dba1513cfe0e5dd85cbf793b3782902b981ac9610b0e472c6a048f738c02`

Exp26 result readback:

- search-dev Step50 minus Step0: whole PSNR `+4.816168`, strict mask PSNR
  `+4.942246`, boundary PSNR `+12.111889`, LPIPS `-0.044059`,
  Ewarp `-7.055122`.
- shadow-dev status: `VIDEOPAINTER_SHADOWDEV_CONFIRMED`.
- shadow-dev Step50 minus Step0: strict mask PSNR `+5.186942`, boundary PSNR
  `+12.175098`, whole PSNR `+5.160739`, LPIPS `-0.040142`,
  Ewarp `-8.378847`.
- external DAVIS-derived status: `EXP26_EXTERNAL_VALIDATION_NOT_CONFIRMED`.

The 50-step run is useful cross-backbone micro evidence, but not a full
long-run trajectory. Exp31 will pre-register Step2000 as the primary endpoint
and keep search-dev trajectory analysis separate from shadow-dev confirmation.

## Right-Side Protection

Read-only PAI checks at `2026-06-27T12:56:38+08:00` and
`2026-06-27T12:58:03+08:00` found no active compute processes on GPUs 0-7.
A targeted read-only check at `2026-06-27T13:04:42+08:00` found no active
Exp30/MiniMax process and no active compute process.

Conservative reservation remains in force:

- Exp30 worktree detected locally at
  `/home/hj/H20_Video_inpainting_DPO_exp30_vor_or_minimax`.
- Exp30/MiniMax output roots exist on PAI.
- stale MiniMax candidate locks are present for GPU0 and GPU5 under
  `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp29_or_adapter_feasibility/`.

Therefore Exp31 may use GPU1 by plan, while GPU0 and GPU5 remain reserved for
right-side or stale-lock protection unless the user explicitly releases them.

## Milestones

1. `EXP31_READBACK_COMPLETED`
2. `VIDEOPAINTER_2000_RESUME_POLICY_AUDITED`
3. `VIDEOPAINTER_2000_L0_L1_PASSED`
4. `VIDEOPAINTER_2000_TRAINING_COMPLETED`
5. `VIDEOPAINTER_2000_EVALUATION_COMPLETED`

The 2000-step run must not start until resume policy and L0/L1 pass.

## Status

Current status: `VIDEOPAINTER_2000_L0_L1_PASSED`.

Resume-policy decision:

- Step50 optimizer state exists.
- Step50 scheduler state is absent.
- Step50 RNG state is absent.
- The 2000-step run must start fresh from Step0 and must not be described as a
  continuation from Step50.

L0/L1 decision:

- run id: `exp31_vp_l0_l1_20260627_132158`
- CUDA_VISIBLE_DEVICES: `1`
- L0 loss: `0.695064902305603`
- L0 policy grad norm: `14.379269412062548`
- reference gradient: `false`
- L1 policy delta norm: `1.6732703166152714`
- L1 reference delta norm: `0.0`
- strict reload max abs diff: `0.0`
- decision: `VIDEOPAINTER_2000_L0_L1_PASSED`
