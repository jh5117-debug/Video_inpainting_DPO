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
8. `VIDEOPAINTER_2000_STRICT_READBACK_COMPLETE_BASE_AUDIT_PENDING`
9. `VIDEOPAINTER_BASE_IDENTITY_AUDIT_PASSED`
10. `VIDEOPAINTER_2000_POSITIVE`

The 2000-step run must not start until resume policy and L0/L1 pass.

## Status

Current status: `VIDEOPAINTER_2000_POSITIVE`.

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

- final status: `VIDEOPAINTER_2000_POSITIVE`
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

The result is now promoted to `VIDEOPAINTER_2000_POSITIVE` for VideoPainter
only because the strict official-base identity audit passed and the completed
LPIPS / mask-region Ewarp gate favors Step2000 over both Step0 and Step50 on
fixed search-dev and shadow-dev. The allowed paper wording is VideoPainter
2000-step long-run positive evidence under this fixed protocol, with the prior
50-step result retained as micro evidence. Universal-adapter, final-SOTA,
all-models-supported, and top-conference-novelty claims remain forbidden.

Strict validation readback:

- status: `VIDEOPAINTER_2000_STRICT_READBACK_COMPLETE`
- report: `reports/exp31_vp_2000_strict_readback.md`
- current finding: source/config readback supports same official base family,
  same search/shadow rows, same 49F protocol, same seed, same mask polarity, and
  same diagnostic comp formula for Step0/50/2000.

Official base identity audit:

- status: `VIDEOPAINTER_BASE_IDENTITY_AUDIT_PASSED`
- report: `reports/exp31_vp_2000_base_identity_audit.md`
- csv: `reports/exp31_vp_2000_base_identity_audit.csv`
- replay diff: `reports/exp31_vp_2000_replay_diff.csv`
- comp formula audit: `reports/exp31_vp_2000_comp_formula_audit.csv`
- summary json: `reports/exp31_vp_2000_base_identity_summary.json`
- replay run root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp31_videopainter_2000step_longrun/exp31_vp2000_base_identity_replay_20260628_091019`
- official base and Step0 branch weights match exactly:
  `5d01728cb0cb605b591f41cbea033db22d5ae72d0b37565957feae71b089be8e`.
- Step50 weight SHA:
  `3849eafbeb9f30a7fb0f52df4c5f0a172d4d437e4161a182a075e15699b2430b`.
- Step2000 weight SHA:
  `fd02a22088da6869fafed437284287b011181882943f81d5ec8b1a493472c148`.
- 2 search-dev and 2 shadow-dev rows were replayed for official base, Step0,
  Step50, and Step2000 using the same 49F/720x480/seed-20260627 protocol.
- official base vs replay Step0, official base vs existing Step0, and
  replay Step0/50/2000 vs existing Step0/50/2000 were all exact:
  `MAE=0`, `max_abs=0`, 49/49 frames, hash-equal raw and comp frames.
- recomputed comp frames exactly match
  `comp = raw inside mask + winner outside mask`, with mask threshold
  `>127` and first-frame mask zeroing; `comp_recalc_mae=0` and first-frame
  mask sum `0` for every checked row.
- no MiniMax, EffectErase adapter, shared trainer, or `inference/metrics.py`
  change was made by the audit.
- formal-positive blocker removed by LPIPS/Ewarp completion.

LPIPS/Ewarp completion:

- status: `VIDEOPAINTER_2000_POSITIVE`
- report: `reports/exp31_vp_2000_lpips_ewarp_metrics.md`
- aggregate CSV: `reports/exp31_vp_2000_lpips_ewarp_metrics.csv`
- per-video CSV: `reports/exp31_vp_2000_lpips_ewarp_per_video.csv`
- paired deltas CSV: `reports/exp31_vp_2000_lpips_ewarp_paired_deltas.csv`
- summary JSON: `reports/exp31_vp_2000_lpips_ewarp_summary.json`
- run root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp31_videopainter_2000step_longrun/exp31_vp2000_lpips_ewarp_20260628_095055`
- rows: `384/384` OK across search-dev and shadow-dev, Step0/50/2000,
  raw/comp.
- shadow-dev comp Step2000 vs Step0: full PSNR `+11.440561`, full LPIPS
  `-0.056840`, mask LPIPS `-0.213718`, boundary PSNR `+15.242894`, mask-region
  Ewarp `-11.171650`, probability improved `1.0000`.
- shadow-dev comp Step2000 vs Step50: full PSNR `+2.305730`, full LPIPS
  `-0.008813`, mask LPIPS `-0.034082`, boundary PSNR `+3.637059`, mask-region
  Ewarp `-0.258536`, probability improved `>=0.9062`.
- TC caveat: `TC_BACKEND_NOT_LOCAL`; no automatic model download and no proxy
  reported as real TC.
- Ewarp caveat: mask-region Ewarp uses the existing `inference/metrics.py`
  backend with OpenCV DIS fallback because RAFT weights were not local on PAI.
- Outside caveat: comp outside pixels are copied from the winner, so outside L1
  is exactly `0.0` by construction and is not model-predicted outside
  preservation evidence.
