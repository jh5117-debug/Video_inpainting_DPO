# PRD 21: Exp11 Region Boundary Ablation and Exp12 Adaptive Normalization

Date: 2026-06-12

## Scope

This PRD defines the next PAI experiment batch. It does not modify old Exp9 / Exp10 / Exp11-proxy code and does not modify shared `training/dpo/train_stage1.py` or `training/dpo/train_stage2.py`.

New experiment folders:

- `exp11_region_boundary_ablation/`
- `exp12_adaptive_normalization/`

New registries:

- `experiment_registry/exp11_region_boundary_ablation/`
- `experiment_registry/exp12_adaptive_normalization/`

## Fixed Setting

All runs keep the Exp10 baseline setting:

- win: GT clean video
- lose: generated rollout loser
- task: partial-mask video inpainting
- training source manifest: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4/manifests/selected_primary_comp.repaired.pai_paths.jsonl`
- effective GT-win manifest: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/exp09_10_11_youtubevos_gtwin_d3comp_pai/manifests/selected_primary_comp.gtwin.pai_paths.jsonl`
- SFT-48000 weight: `/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000`
- prior mode: ProPainter prior
- DAVIS eval root: `/mnt/workspace/hj/nas_hj/data/external/davis_432_240`
- eval protocol: DAVIS50 raw6 hard-comp / D+G off / no PCM / frame-wise in-memory metric
- canonical metric wrapper: `tools/run_davis50_framewise_protocol_eval.py`
- inpainting metrics: project metric code only; no VBench
- validation outputs: metrics summaries, four-column side-by-side videos, index, pair manifest, report

Important correction from 2026-06-14:

- Final metric tables must not use the generic saved-output pair-manifest metric
  path.
- The canonical path is the DAVIS50 raw6 hard-comp frame-wise in-memory wrapper.
- D+G off means no mask dilation and no Gaussian blur during comp.
- The expected score regime for SFT-48000 is 32+ PSNR under this protocol.

## Why Exp11

Exp10 changed the loss from full-frame MSE to region-local MSE. The next ambiguity is the boundary definition:

- `inner`: mask-side ring, `mask - erode(mask)`
- `outer`: outside-context ring, `dilate(mask) - mask`
- `both`: `dilate(mask) - erode(mask)`

Exp11 tests whether the boundary term should emphasize the hole-side edge, the context-side edge, or both. It also tests whether boundary weight `0.75` or `1.0` is more stable.

Exp11 variants:

- `exp11_boundary_inner_b075_o005_s1s2_2000`
- `exp11_boundary_outer_b075_o005_s1s2_2000`
- `exp11_boundary_both_b075_o005_s1s2_2000`
- `exp11_boundary_both_b100_o005_s1s2_2000`

## Why Exp12

Exp9 log-ratio is reference-level normalization:

```text
g_w = log((m_w + eps) / (m_w_ref + eps))
g_l = log((m_l + eps) / (m_l_ref + eps))
```

It normalizes absolute error differences into error ratios relative to the frozen SFT reference.

Exp12 adds batch-level adaptive normalization after the log-ratio:

```text
g_w_adapt = (g_w - mean(g_w_batch)) / (std(g_w_batch) + eps)
g_l_adapt = (g_l - mean(g_l_batch)) / (std(g_l_batch) + eps)
```

This tests whether batch z-score normalization stabilizes win-gap / lose-gap distributions better than log-ratio alone.

Launched variant:

- `exp12_batch_adaptive_norm_s1s2_2000`

Prepared but not launched:

- `exp12_timestep_adaptive_norm_s1s2_2000`

## Exp12 Outer Boundary Follow-Up

The original Exp12 launched variant used the Exp10 default boundary setting:

- `boundary_mode=exp10_default`
- `boundary_weight=0.5`

After the canonical DAVIS50 rerun, the best current result is:

- `Exp11_boundary_outer_b075_S2`

To test whether adaptive normalization still helps with this stronger boundary
setting, create an isolated follow-up:

- folder: `exp12_adaptive_outer_boundary/`
- registry: `experiment_registry/exp12_adaptive_outer_boundary/`
- launcher: `scripts/launch_exp12_adaptive_outer_boundary_pai.sh`
- variant: `exp12_batch_adaptive_outer_b075_s1s2_2000`
- boundary setting: `boundary_mode=outer`, `boundary_weight=0.75`

Decision rule:

- If `Exp12 adaptive + outer b0.75` beats `Exp11 outer b0.75 S2`, keep it as
  the next candidate.
- If it does not, keep `Exp11 outer b0.75 S2` as the current best and keep the
  original Exp12 as a normalization ablation.

Final status on 2026-06-15:

- `Exp12 adaptive + outer b0.75` completed Stage1, Stage1 DAVIS eval, Stage2,
  and Stage2 DAVIS eval on PAI.
- It improved over its local SFT-48000 baseline but did not beat
  `Exp11_boundary_outer_b075_S2`.
- Current best remains `Exp11_boundary_outer_b075_S2`.

Canonical DAVIS50 scores:

| Method | Stage | PSNR | SSIM | LPIPS | VFID | TC | Mask PSNR |
|---|---|---:|---:|---:|---:|---:|---:|
| SFT-48000 baseline | SFT base | 32.731391 | 0.970533 | 0.016660 | 0.201792 | 0.971200 | 23.884924 |
| Exp11 boundary outer b0.75 | DPO-S1 + SFT-S2 | 32.901188 | 0.971859 | 0.015104 | 0.188015 | 0.971287 | 24.054721 |
| Exp11 boundary outer b0.75 | DPO-S1 + DPO-S2 | 33.013954 | 0.972295 | 0.015363 | 0.175423 | 0.971122 | 24.167487 |
| Exp12 adaptive norm | DPO-S1 + DPO-S2 | 32.902760 | 0.972035 | 0.015377 | 0.184785 | 0.970914 | 24.056294 |
| Exp12 adaptive + outer b0.75 | DPO-S1 + DPO-S2 | 32.856975 | 0.971585 | 0.015605 | 0.193578 | 0.971475 | 24.010508 |

Interpretation:

- The most defensible claim is not "adaptive normalization wins"; it does not
  win under the fixed protocol.
- The current best method is the region-local normalized DPO with outer-boundary
  weighting, `boundary_mode=outer`, `boundary_weight=0.75`, full Stage1+Stage2.
- Exp12 remains useful as a negative/ablation result: batch z-score normalization
  did not improve over the simpler log-ratio normalized region-local DPO with
  the best boundary setting.

Artifact archive on HAL:

- dirty pre-cleanup archive:
  `/home/hj/dpo-2-1-exp/local_dirty_archive/H20_hal_dirty_20260615_054311`
- this-week metric and dpo-diag archive:
  `/home/hj/dpo-2-1-exp/this_week_exp11_exp12`
- selected visual evidence run:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/20260615_exp11_outer_b075_s2_selected_visuals`
- selected visual evidence copied to HAL:
  `/home/hj/dpo-2-1-exp/this_week_exp11_exp12/visual_evidence/exp11_outer_b075_s2_selected_visuals`

Selected DAVIS visual candidates were chosen by per-video improvement over
SFT-48000, prioritizing mask-region PSNR / mask-region SSIM / LPIPS. The
2026-06-15 evidence rerun confirmed the strongest positive examples are:

- `boat`
- `rhino`
- `dog-agility`
- `lucia`
- `blackswan`

`boat` is the clearest paper/PPT qualitative example: SFT-48000 produces a
visible white fog / patch over the boat wake and hull, while Exp11 outer b0.75
S2 keeps cleaner water texture and boundary continuity. `rhino` and
`dog-agility` are also useful because the mask crosses foreground object
boundaries. `dance-jump` and `soccerball` should be treated as failure or
cautionary examples in this selected rerun because their per-video PSNR/SSIM
decreased versus SFT-48000.

## DPO Loss

Both experiments inherit Exp10's region-local normalized DPO:

```text
m = sum(region_weight_map * mse_map) / (sum(region_weight_map) + eps)

g_w = log((m_w + eps) / (m_w_ref + eps))
g_l = log((m_l + eps) / (m_l_ref + eps))
g_l_clip = clip(g_l, max=1.0)

L_DPO = mean[-logsigmoid(-0.5 * 10 * (g_w - 0.25 * g_l_clip))]

L_total = L_DPO + 0.05 * m_w + ReLU(g_w)
```

Exp12 replaces `g_w/g_l` in the DPO term and winner-gap term with `g_w_adapt/g_l_adapt` for the launched batch-zscore variant.

## Do Not Do

Do not run OR experiments in this batch. OR is deferred because there is no OR training data and OR metrics are not the same as BR/inpainting metrics.

Do not run DPO adapter experiments on other baselines in this batch. Future work must first check whether the baseline has open training code, diffusion-style reference compatibility, and a viable DPO adapter path.

Do not run Exp11-real flow-prior long training in this batch. The old Exp11 line is proxy consistency, not real RAFT/ProPainter flow-prior consistency.

## Baseline Adapter Follow-Up

The next possible direction is to treat the winning Exp11 outer-boundary loss as
a BR/inpainting adapter objective for other public inpainting baselines. This is
not part of the Exp11/Exp12 training batch and must stay isolated.

Rules:

- Do not copy any baseline into shared training code.
- Each baseline must live under its own isolated folder and registry.
- Use YouTube-VOS for training and DAVIS/DAVIS-test-style evaluation where the
  dataset contract can be matched.
- Keep the fixed DAVIS50 raw6 hard-comp metric protocol for DiffuEraser
  comparisons.
- If a baseline only provides inference code and no training code, do not claim
  a trainable adapter.

Current public-code audit:

- `VideoPainter`: public training entrypoints exist; this is the first viable
  adapter candidate.
- `COCOCO`: training code appears not yet public / under preparation.
- `VACE`: inference/preprocess style code found; no validated training entry.
- `VideoComposer/VideoComp`: partial utilities exist, but no clean train entry
  for this adapter path was validated.
- `FloED`: no reliable public training repository validated yet.

## Launch

Use:

```bash
bash scripts/launch_exp11_exp12_parallel_pai.sh
```

The launcher starts:

- Task A visual case selection for Exp10-2, no GPU.
- Exp11 boundary ablation on GPU `0,1,2,3`, variants sequentially.
- Exp12 adaptive normalization on GPU `4,5,6,7`.

No process killing is allowed; the launcher only uses `CUDA_VISIBLE_DEVICES`.
