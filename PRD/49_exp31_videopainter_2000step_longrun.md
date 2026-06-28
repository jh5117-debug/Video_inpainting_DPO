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
5. `VIDEOPAINTER_2000_STEP0_50_2000_EVAL_RUNNING`
6. `VIDEOPAINTER_2000_EVALUATION_COMPLETED`
7. `VIDEOPAINTER_2000_PARETO_MIXED`

The 2000-step run must not start until resume policy and L0/L1 pass.

## Status

Current status: `VIDEOPAINTER_2000_PARETO_MIXED`.

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

Checkpoint ladder readiness:

- trainer argument: `--checkpoint_steps`
- planned explicit checkpoints:
  `0,1,10,50,100,200,500,1000,1500,2000`
- retention behavior: explicit checkpoint steps are protected from pruning.
- trainer state: optimizer, explicit `lr_scheduler`, and Python/NumPy/Torch/CUDA
  RNG state are saved in `trainer_state.pt`.
- periodic checkpointing may be disabled with `--checkpointing_steps 0` when an
  explicit list is provided.
- validation: `git diff --check`, `py_compile`, 28 unit tests, and `bash -n`.

Training completion readback:

- run id: `exp31_vp2000_fresh_step0_20260627_133831`
- status: completed / `rc=0`
- final step: `2000`
- checkpoint ladder present:
  `0,1,10,50,100,200,500,1000,1500,2000`
- dpo diagnostics rows: `2000`

Evaluation launch:

- eval run id: `exp31_vp2000_eval_step0_50_2000_20260628_032700`
- run root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp31_videopainter_2000step_longrun/exp31_vp2000_eval_step0_50_2000_20260628_032700`
- GPU: `GPU1`
- controller PID/PGID at launch check: `1945068/1945066`
- checkpoints: `step0`, `step50`, `step2000`
- splits: fixed `search-dev` and fixed `shadow-dev`
- no new training is launched by this evaluation controller.
- no MiniMax / Exp36 / GPU0 work is launched or modified.

Evaluation completion:

- eval run id: `exp31_vp2000_eval_step0_50_2000_20260628_032700`
- completed checkpoints: `step0`, `step50`, `step2000`
- completed splits: fixed `search-dev` and fixed `shadow-dev`
- all 6 generation/review groups completed with `32/32` valid rows.
- external status: `VIDEOPAINTER_2000_EXTERNAL_NOT_AVAILABLE`; no Exp31-specific
  external Step0/Step2000 evaluation was fabricated or reused from Exp26.

Final decision:

- final status: `VIDEOPAINTER_2000_PARETO_MIXED`
- search-dev Step2000 vs Step0: full PSNR `+5.5701`, mask PSNR `+9.9747`,
  sampled boundary PSNR `+12.0920`, win rate `0.9688`.
- search-dev Step2000 vs Step50: full PSNR `+6.1338`, mask PSNR `+1.8747`,
  sampled boundary PSNR `+3.7226`, sampled outside L1 `-10.0351`, win rate
  `1.0000`.
- shadow-dev Step2000 vs Step0: full PSNR `+6.2632`, mask PSNR `+10.8860`,
  sampled boundary PSNR `+12.2343`, win rate `1.0000`.
- shadow-dev Step2000 vs Step50: full PSNR `+6.4772`, mask PSNR `+2.0832`,
  sampled boundary PSNR `+3.9405`, sampled outside L1 `-10.5232`, win rate
  `1.0000`.

Video review covered the all-32 evidence/crop pages for search-dev and
shadow-dev at Step0, Step50, and Step2000. Step2000 is visibly cleaner than
Step50 and not collapsed; Step50 has repeated outside brightness/color
pollution that is much reduced at Step2000. A minority of Step2000 rows still
show finite residual local texture or mild darkening.

The result is deliberately not promoted to `VIDEOPAINTER_2000_POSITIVE` because
LPIPS and Ewarp were not computed in this fast summary, and the prompt requires
those metrics for a formal positive gate. The allowed paper wording is
qualified 2000-step long-run evidence, with the prior 50-step result retained as
micro evidence. Universal-adapter, final-SOTA, all-models-supported, and
top-conference-novelty claims remain forbidden.
